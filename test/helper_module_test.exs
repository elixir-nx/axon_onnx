defmodule HelperText do
  use ExUnit.Case, async: true

  alias Onnx.AttributeProto, as: Attribute
  alias Onnx.TensorProto, as: Placeholder
  alias Onnx.ValueInfoProto, as: Value
  alias Onnx.TypeProto, as: Type
  alias Onnx.NodeProto, as: Node
  alias Onnx.ModelProto, as: Model
  alias Onnx.TensorShapeProto.Dimension
  alias AxonOnnx.Helper

  @cache_dir Path.join([File.cwd!(), ".test-cache"])

  describe "Creates new tensor" do
    test "Float 16 Tensor with {1,2,3} shape with bitcast to unsigned int" do
      assert %Placeholder{
               name: "float16_tensor_test",
               int32_data: [18816, 16486, 16896, 17434, 17792, 18099],
               dims: [1, 2, 3],
               data_type: 10
             } =
               Helper.make_tensor(
                 "float16_tensor_test",
                 :FLOAT16,
                 {1, 2, 3},
                 [11.0, 2.2, 3.0, 4.1, 5.5, 6.7],
                 false
               )
    end

    test "Float Tensor with {1,2,3} shape" do
      assert %Placeholder{
               name: "float_tensor_test",
               float_data: [11.0, 2.2, 3.0, 4.1, 5.5, 6.7],
               dims: [1, 2, 3],
               data_type: 1
             } =
               Helper.make_tensor(
                 "float_tensor_test",
                 :FLOAT,
                 {1, 2, 3},
                 [11.0, 2.2, 3.0, 4.1, 5.5, 6.7],
                 false
               )
    end

    test "Wrong numeric format Tensor" do
      assert_raise ArgumentError, ~r/Wrong data_type format. Expected atom or number/, fn ->
        Helper.make_tensor("wrong_tensor_test", 99, {1, 2, 3}, [117], false)
      end
    end

    test "Wrong atom format Tensor" do
      assert_raise ArgumentError, ~r/Wrong data_type format. Expected atom or number/, fn ->
        Helper.make_tensor("wrong_tensor_test", :WRONG, {1, 2, 3}, [117], false)
      end
    end

    test "Wrong number of values" do
      assert_raise ArgumentError, ~r/Number of values does not match tensor's size./, fn ->
        Helper.make_tensor("float_tensor_test", :FLOAT, {1, 2, 3}, [11.0], false)
      end
    end

    test "Wrong string format Tensor with raw data" do
      assert_raise ArgumentError, ~r/Can not use raw_data to store string type/, fn ->
        Helper.make_tensor("wrong_tensor_test", :STRING, {1}, ["test string"], true)
      end
    end
  end

  describe "Creates new ValueInfoProto" do
    test "" do
      %Value{
        name: "033_convolutional_conv_weights",
        type: %Type{
          value: {:tensor_type, type_proto_t}
        }
      } = Helper.make_tensor_value_info("033_convolutional_conv_weights", 1, {128, 256, 1, 1})

      assert type_proto_t.elem_type === 1
      assert %Dimension{value: {:dim_value, 128}} = Enum.at(type_proto_t.shape.dim, 0)
      assert %Dimension{value: {:dim_value, 1}} = Enum.at(type_proto_t.shape.dim, 3)
    end
  end

  describe "Creates new attribute" do
    test "Single float attribute" do
      kwargs = %{alpha: 0.1}

      assert [%Attribute{f: 0.1, type: :FLOAT, name: "alpha", floats: [], i: 0}] =
               Helper.make_attribute(kwargs)
    end

    test "Multiple list of integers and string attributes" do
      kwargs = %{auto_pad: "SAME_LOWER", dilations: [1, 1]}

      assert [
               %Attribute{name: "dilations", ints: [1, 1], type: :INTS},
               %Attribute{name: "auto_pad", s: "SAME_LOWER", type: :STRING}
             ] = Helper.make_attribute(kwargs)
    end
  end

  describe "Create new node" do
    test "New node with single input and single output without attributes" do
      assert %Node{
               op_type: "Relu",
               input: ["X"],
               output: ["Y"]
             } = Helper.make_node("Relu", ["X"], ["Y"])
    end

    test "New node with single input and single output with attributes" do
      %Node{
        attribute: [
          %Attribute{
            i: 1,
            name: "arg_value",
            type: :INT
          }
        ],
        input: ["X"],
        name: "",
        op_type: "Relu",
        output: ["Y"]
      } = Helper.make_node("Relu", ["X"], ["Y"], "", arg_value: 1)
    end
  end

  describe "Create new graph" do
    test "" do
      node_def1 = Helper.make_node("Relu", ["X"], ["Y"])
      node_def2 = Helper.make_node("Add", ["X", "Y"], ["Z"])
      value_info = [Helper.make_tensor_value_info("Y", :FLOAT, [1, 2])]

      graph =
        Helper.make_graph(
          [node_def1, node_def2],
          "test",
          [Helper.make_tensor_value_info("X", :FLOAT, [1, 2])],
          [Helper.make_tensor_value_info("Z", :FLOAT, [1, 2])],
          [],
          "",
          value_info
        )

      assert graph.name === "test"
      assert Enum.count(graph.node) == 2
      assert Enum.at(graph.node, 0) === node_def1
      assert Enum.at(graph.node, 1) === node_def2
      assert graph.doc_string === ""
      assert Enum.at(graph.value_info, 0) === Enum.at(value_info, 0)
    end
  end

  describe "Create and export new model" do
    def create_graph() do
      Helper.make_graph(
        [Helper.make_node("Relu", ["X"], ["Y"])],
        "input_test",
        [Helper.make_tensor_value_info("X", :FLOAT, [1, 2])],
        [Helper.make_tensor_value_info("Y", :FLOAT, [1, 2])],
        [
          Helper.make_tensor("init_test", :FLOAT, {1}, [1]),
          Helper.make_tensor("X", :FLOAT, {2}, [1, 2])
        ]
      )
    end

    test "Create new model" do
      graph_def = create_graph()

      assert %Model{
               producer_name: "test"
             } = Helper.make_model(graph_def, producer_name: "test")
    end

    test "Remove input layers from initializer" do
      graph_def = create_graph()
      m = Helper.make_model(graph_def, producer_name: "optimizer_test")
      opt_model = Helper.remove_initializer_from_input(m)
      assert opt_model.graph.input === []
      assert m.graph.initializer === opt_model.graph.initializer
    end

    test "Save the new model and check if the file has been saved" do
      model_name = "temporary_model_test"
      cache_dir = Path.join([@cache_dir, model_name])
      File.mkdir_p!(cache_dir)

      model_path = Path.join([cache_dir, "#{model_name}.onnx"])
      graph_def = create_graph()

      model_def =
        %Model{
          producer_name: "test"
        } = Helper.make_model(graph_def, producer_name: "test")

      assert :ok = Helper.save_model(model_def, model_path)

      assert cache_dir
             |> File.ls!()
             |> Enum.find(fn x -> x === "#{model_name}.onnx" end)

      assert File.rm!(model_path)
    end
  end
end
