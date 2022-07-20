defmodule AxonOnnx.Helper do
  @moduledoc """
    Helper class used for creating tensors
    (partially ported from: https://github.com/onnx/onnx/blob/master/onnx/helper.py)
  """
  use Agent, restart: :transient

  @onnx_opset_version 15
  @onnx_ir_version 8

  alias AxonOnnx.Mapping
  alias Onnx.ModelProto, as: Model
  alias Onnx.GraphProto, as: Graph
  alias Onnx.NodeProto, as: Node
  alias Onnx.ValueInfoProto, as: Value
  alias Onnx.AttributeProto, as: Attribute
  alias Onnx.OperatorSetIdProto, as: Opset
  alias Onnx.TypeProto, as: Type
  alias Onnx.TensorProto, as: Placeholder
  alias Onnx.TypeProto.Tensor, as: TensorTypeProto
  alias Onnx.TensorShapeProto, as: Shape
  alias Onnx.TensorShapeProto.Dimension, as: Dimension

  # Checks whether a variable is enumerable and not a struct
  defp is_enum?(var) do
    is_list(var) or
      (is_map(var) and not Map.has_key?(var, :__struct__)) or
      is_tuple(var)
  end

  defp data_type_id_from_atom(data_type) when is_atom(data_type) do
    # Get the data_type number from atom
    Enum.find(Placeholder.DataType.constants(), fn {n, t} ->
      t == data_type && n
    end)
  end

  @doc """
    Construct an OperatorSetIdProto.
    Arguments:
        domain (string): The domain of the operator set id
        version (integer): Version of operator set id
  """
  def make_operatorsetid(domain, version) do
    %Opset{
      domain: domain,
      version: version
    }
  end

  defp parse_data_type(data_type) do
    parsed_data_type =
      cond do
        is_atom(data_type) ->
          # Check for an existing type identified by the atom
          data_type_id_from_atom(data_type)

        is_number(data_type) ->
          # Check for an existing type identified by the number
          Enum.fetch(Placeholder.DataType.constants(), data_type)

        true ->
          nil
      end

    case parsed_data_type do
      {:ok, dt} ->
        dt

      {n, a} when is_number(n) and is_atom(a) ->
        parsed_data_type

      _ ->
        max_data_type_id = Enum.count(Placeholder.DataType.constants()) - 1

        raise ArgumentError,
              "Wrong data_type format. Expected atom or number<#{max_data_type_id}, got: #{data_type}"
    end
  end

  @doc """
    Make a TensorProto with specified arguments.  If raw is False, this
    function will choose the corresponding proto field to store the
    values based on data_type. If raw is True, use "raw_data" proto
    field to store the values, and values should be of type bytes in
    this case.
  """
  def make_tensor(name, data_type, dims, vals, raw \\ false) do
    {data_type_id, data_type_atom} = parse_data_type(data_type)

    if data_type_id == 8 and raw == true,
      do: raise(ArgumentError, "Can not use raw_data to store string type")

    itemsize = Mapping.tensor_type_to_nx_size()[data_type_atom]
    expected_size = (raw == false && 1) || itemsize
    expected_size = Enum.reduce(Tuple.to_list(dims), expected_size, fn val, acc -> acc * val end)

    if Enum.count(vals) != expected_size,
      do:
        raise(
          ArgumentError,
          "Number of values does not match tensor's size. Expected #{expected_size}, but it is #{Enum.count(vals)}. "
        )

    tensor = %Placeholder{
      data_type: data_type_id,
      name: name,
      # raw_data: (raw && vals) || "",
      # float_data: (!raw && vals) || [],
      dims: Tuple.to_list(dims)
    }

    # TODO @stefkohub add support for complex values
    if raw == true do
      %{tensor | raw_data: vals}
    else
      tvalue =
        cond do
          # float16/bfloat16 are stored as uint16
          data_type_atom == :FLOAT16 or data_type_atom == :BFLOAT16 ->
            Nx.tensor(vals, type: {:f, 16})
            |> Nx.bitcast({:u, 16})
            |> Nx.to_flat_list()

          data_type_atom != :COMPLEX64 and data_type_atom != :COMPLEX128 ->
            vals

          true ->
            raise ArgumentError, "Unsupported data type: #{data_type_atom}"
        end

      Map.replace(
        tensor,
        Mapping.storage_tensor_type_to_field()[
          Mapping.tensor_type_atom_to_storage_type()[data_type_atom]
        ],
        tvalue
      )
    end
  end

  @doc """
    Create a ValueInfoProto structure with internal TypeProto structures
  """
  def make_tensor_value_info(name, elem_type, shape, doc_string \\ "", shape_denotation \\ "") do
    {elem_type_id, _elem_type_atom} = parse_data_type(elem_type)
    the_type = make_tensor_type_proto(elem_type_id, shape, shape_denotation)

    %Value{
      name: name,
      doc_string: (doc_string !== "" && doc_string) || "",
      type: the_type
    }
  end

  @doc """
    Create a TypeProto structure to be used by make_tensor_value_info
  """
  def make_tensor_type_proto(elem_type, shape, shape_denotation \\ []) do
    %Type{
      value:
        {:tensor_type,
         %TensorTypeProto{
           elem_type: elem_type,
           shape:
             if shape != nil do
               if is_enum?(shape_denotation) == true and Enum.count(shape_denotation) != 0 and
                    Enum.count(shape_denotation) != tuple_size(shape) do
                 raise "Invalid shape_denotation. Must be the same length as shape."
               end

               %Shape{dim: create_dimensions(shape, shape_denotation)}
             else
               %Shape{}
             end
         }}
    }
  end

  # Create a TensorShapeProto.Dimension structure based on shape types
  defp create_dimensions(shape, shape_denotation) do
    list_shape = (is_tuple(shape) && Tuple.to_list(shape)) || shape

    list_shape
    |> Enum.with_index()
    |> Enum.map(fn {acc, index} ->
      cond do
        is_integer(acc) ->
          %Dimension{
            value: {:dim_value, acc},
            denotation:
              if shape_denotation != "" do
                Enum.at(shape_denotation, index)
              else
                ""
              end
          }

        is_binary(acc) ->
          %Dimension{
            value: {:dim_param, acc},
            denotation:
              if shape_denotation != "" do
                Enum.at(shape_denotation, index)
              else
                ""
              end
          }

        [] ->
          _ = IO.puts("Empty acc")

        true ->
          raise "Invalid item in shape: #{inspect(acc)}. Needs to be integer or text type."
      end
    end)
    |> List.flatten()
  end

  @doc """
    Creates a GraphProto 
  """
  def make_graph(
        nodes,
        name,
        inputs,
        outputs,
        initializer \\ [],
        doc_string \\ "",
        value_info \\ [],
        sparse_initializer \\ []
      ) do
    %Graph{
      doc_string: doc_string,
      initializer: initializer,
      input: inputs,
      name: name,
      node: nodes,
      output: outputs,
      quantization_annotation: [],
      sparse_initializer: sparse_initializer,
      value_info: value_info
    }
  end

  @doc """
    Creates a ModelProto
  """
  def make_model(graph, kwargs) do
    %Model{
      doc_string: Keyword.get(kwargs, :doc_string, ""),
      domain: Keyword.get(kwargs, :domain, ""),
      graph: graph,
      ir_version: @onnx_ir_version,
      metadata_props: Keyword.get(kwargs, :metadata_props, []),
      model_version: Keyword.get(kwargs, :model_version, 1),
      opset_import:
        Keyword.get(kwargs, :opset_imports, [%Opset{domain: "", version: @onnx_opset_version}]),
      producer_name: Keyword.get(kwargs, :producer_name, ""),
      producer_version: Keyword.get(kwargs, :producer_version, "0.0.1-sf"),
      training_info: Keyword.get(kwargs, :training_info, [])
    }
  end

  @doc """
    Prints a high level representation of a GraphProto
  """
  def printable_graph(graph) do
    IO.puts("============================================================")
    IO.puts("            Graph: " <> graph.name)
    IO.puts("     Output nodes: ")

    Enum.each(graph.output, fn o ->
      dims = for d <- elem(o.type.value, 1).shape.dim, do: elem(d.value, 1)
      IO.puts("        " <> o.name <> " " <> inspect(dims))
    end)

    IO.puts("============================================================")
  end

  @doc """
    Encodes and write a binary file f containing a ModelProto
  """
  def save_model(proto, f) do
    encoded_model = Onnx.ModelProto.encode!(proto)
    {:ok, file} = File.open(f, [:write])
    IO.binwrite(file, encoded_model)
    File.close(file)
  end

  @doc """
    Helper functions checking whether the passed val is of any of the Onnx types 
  """
  def is_TensorProto(val) do
    is_map(val) and Map.has_key?(val, :__struct__) and
      val.__struct__ === Onnx.TensorProto
  end

  def is_SparseTensorProto(val) do
    is_map(val) and Map.has_key?(val, :__struct__) and
      val.__struct__ === Onnx.SparseTensorProto
  end

  def is_GraphProto(val) do
    is_map(val) and Map.has_key?(val, :__struct__) and
      val.__struct__ === Onnx.GraphProto
  end

  def is_TypeProto(val) do
    is_map(val) and Map.has_key?(val, :__struct__) and
      val.__struct__ === Onnx.TypeProto
  end

  defp create_attribute_map(key, val) do
    to_add =
      cond do
        is_float(val) ->
          %{f: val, type: :FLOAT}

        is_integer(val) ->
          %{i: val, type: :INT}

        is_binary(val) or is_boolean(val) ->
          %{s: val, type: :STRING}

        is_TensorProto(val) ->
          %{t: val, type: :TENSOR}

        is_SparseTensorProto(val) ->
          %{sparse_tensor: val, type: :SPARSE_TENSOR}

        is_GraphProto(val) ->
          %{g: val, type: :GRAPH}

        is_TypeProto(val) ->
          %{tp: val, type: :TYPE_PROTO}

        is_enum?(val) && Enum.all?(val, fn x -> is_integer(x) end) ->
          %{ints: val, type: :INTS}

        is_enum?(val) and Enum.all?(val, fn x -> is_float(x) or is_integer(x) end) ->
          # Convert all the numbers to float
          %{floats: Enum.map(val, fn v -> v / 1 end), type: :FLOATS}

        is_enum?(val) and Enum.all?(val, fn x -> is_binary(x) end) ->
          %{strings: val, type: :STRINGS}

        is_enum?(val) and Enum.all?(val, fn x -> is_TensorProto(x) end) ->
          %{tensors: val, type: :TENSORS}

        is_enum?(val) and Enum.all?(val, fn x -> is_SparseTensorProto(x) end) ->
          %{sparse_tensors: val, type: :SPARSE_TENSORS}

        is_enum?(val) and Enum.all?(val, fn x -> is_GraphProto(x) end) ->
          %{graphs: val, type: :GRAPHS}

        is_enum?(val) and Enum.all?(val, fn x -> is_TypeProto(x) end) ->
          %{type_protos: val, type: :TYPE_PROTOS}
      end

    Map.merge(
      %Attribute{
        name: Atom.to_string(key)
      },
      to_add
    )
  end

  @doc """
    Creates an attribute based on passed kwargs
  """
  def make_attribute(kwargs) do
    sortedargs = for {k, v} <- Enum.sort(kwargs), v != "", do: {k, v}

    Enum.reduce(sortedargs, [], fn {key, val}, acc ->
      [create_attribute_map(key, val) | acc]
    end)
  end

  @doc """
        Construct a NodeProto.
        Arguments:
        op_type (string): The name of the operator to construct
        inputs (list of string): list of input names
        outputs (list of string): list of output names
        name (string, default None): optional unique identifier for NodeProto
        doc_string (string, default None): optional documentation string for NodeProto
        domain (string, default None): optional domain for NodeProto.
            If it's None, we will just use default domain (which is empty)
        kwargs (dict): the attributes of the node.  The acceptable values
            are documented in :func:`make_attribute`.
  """
  def make_node(
        op_type,
        inputs,
        outputs,
        name \\ "",
        kwargs \\ [],
        doc_string \\ "",
        domain \\ ""
      ) do
    %Node{
      op_type: op_type,
      input: inputs,
      output: outputs,
      name: name,
      domain: domain,
      doc_string: doc_string,
      attribute: make_attribute(kwargs)
    }
  end

  def remove_initializer_from_input(%Model{graph: graph} = model) do
    %{
      model
      | graph: %Graph{
          graph
          | input:
              Enum.reject(model.graph.input, fn input ->
                Enum.find(model.graph.initializer, fn init -> input.name === init.name end)
              end)
        }
    }
  end
end
