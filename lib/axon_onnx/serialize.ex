defmodule AxonOnnx.Serialize do
  @moduledoc false

  alias Onnx.ModelProto, as: Model
  alias Onnx.GraphProto, as: Graph
  alias Onnx.NodeProto, as: Node
  alias Onnx.ValueInfoProto, as: Value
  alias Onnx.AttributeProto, as: Attribute
  alias Onnx.OperatorSetIdProto, as: Opset
  alias Onnx.TypeProto, as: Type
  alias Onnx.TypeProto.Tensor, as: Placeholder
  alias Onnx.TensorProto, as: Tensor
  alias Onnx.TensorShapeProto, as: Shape
  alias Onnx.TensorShapeProto.Dimension, as: Dimension

  import AxonOnnx.Shared

  @onnx_ir_version 3
  @onnx_opset_version 13
  @producer_name "AxonOnnx"
  @producer_version "0.3.0"

  def __dump__(%Axon{} = axon, inputs, params, opts) do
    %Model{graph: %Graph{name: output_name}} =
      onnx_model = to_onnx_model(axon, inputs, params, opts)

    {Model.encode!(onnx_model), output_name}
  end

  defp to_onnx_model(axon, inputs, params, opts) do
    model_version = opts[:version] || 1
    doc_string = opts[:doc_string] || "An Axon Model"

    opset = %Opset{domain: "", version: @onnx_opset_version}

    graph = to_onnx_graph(axon, inputs, params)

    %Model{
      ir_version: @onnx_ir_version,
      producer_name: @producer_name,
      producer_version: @producer_version,
      domain: "",
      model_version: model_version,
      doc_string: doc_string,
      graph: graph,
      opset_import: [opset]
    }
  end

  defp to_onnx_graph(
         %Axon{output: id, nodes: nodes} = axon,
         templates,
         params_or_initializers
       ) do
    %Axon.Node{op: op, name: output_name_fn} = output_node = nodes[id]

    {inputs, param_names, nodes, op_counts, cache} =
      to_onnx(output_node, nodes, templates, [], [], [], %{}, %{})

    output_name =
      case cache do
        %{^id => name} ->
          name

        %{} ->
          output_name_fn.(op, op_counts)
      end

    output_shape = Axon.get_output_shape(axon, templates)

    # Flatten params_or_initializers so it's no longer nested
    # TODO: This is going to be expensive, find a better way
    params_or_initializers =
      params_or_initializers
      |> Enum.reduce(%{}, fn {layer_name, params}, acc ->
        params
        |> Enum.reduce(acc, fn {param_name, v}, acc ->
          Map.put(acc, layer_name <> "_" <> param_name, v)
        end)
      end)

    # Building the initializers with Tensors will result in a bunch of expensive
    # copies, so we instead accumulate names and then use them to build initializers
    # later
    initializers = to_initializers(params_or_initializers, param_names)

    # Parameters need to be specified as graph inputs as well
    updated_inputs =
      param_names
      |> Enum.reduce(
        inputs,
        fn x, acc ->
          param_value = to_value_info(x, Nx.shape(params_or_initializers[x]))
          [param_value | acc]
        end
      )

    {output, _, _} = to_value_info(output_node, output_shape, op_counts, cache)

    %Graph{
      node: Enum.reverse(nodes),
      name: output_name,
      input: updated_inputs,
      output: [output],
      initializer: initializers
    }
  end

  defp to_onnx(
         %Axon.Node{id: id, op: :constant, name: name, opts: [value: v]},
         _nodes_map,
         _templates,
         inputs,
         param_names,
         nodes,
         op_counts,
         cache
       ) do
    name = name.(:constant, op_counts)
    op_counts = Map.update(op_counts, :constant, 1, fn x -> x + 1 end)
    cache = Map.put(cache, id, name)

    value_tensor = to_tensor_proto(v)
    value_attr = to_attr("value", :TENSOR, value_tensor)

    node = %Node{
      input: [],
      output: [name],
      name: name,
      op_type: "Constant",
      attribute: [value_attr]
    }

    {inputs, param_names, [node | nodes], op_counts, cache}
  end

  defp to_onnx(
         %Axon.Node{op: :input, name: name} = axon,
         _nodes_map,
         templates,
         inputs,
         param_names,
         nodes,
         op_counts,
         cache
       ) do
    # TODO: Handle defaults
    name = name.(:input, op_counts)

    shape =
      case templates do
        %Nx.Tensor{} = tensor ->
          Nx.shape(tensor)

        map ->
          Nx.shape(map[name])
      end

    {input_value, op_counts, cache} = to_value_info(axon, shape, op_counts, cache)
    {[input_value | inputs], param_names, nodes, op_counts, cache}
  end

  ## Linear

  defp to_onnx(
         %Axon.Node{
           id: id,
           op: :dense,
           name: name_fn,
           parent: [inp_id],
           parameters: params
         },
         nodes_map,
         templates,
         inputs,
         param_names,
         nodes,
         op_counts,
         cache
       ) do
    {inputs, param_names, nodes, op_counts, cache} =
      to_onnx(
        nodes_map[inp_id],
        nodes_map,
        templates,
        inputs,
        param_names,
        nodes,
        op_counts,
        cache
      )

    inp_name = cache[inp_id]

    {name, op_counts, cache} =
      case cache do
        %{^id => name} ->
          {name, op_counts, cache}

        %{} ->
          name = name_fn.(:dense, op_counts)
          op_counts = Map.update(op_counts, :dense, 1, fn x -> x + 1 end)
          cache = Map.put(cache, id, name)
          {name, op_counts, cache}
      end

    updated_param_names =
      Enum.map(params, fn %{name: p_name} ->
        name <> "_" <> p_name
      end)

    node = %Node{
      input: [inp_name | updated_param_names],
      output: [name],
      name: name,
      op_type: "Gemm"
    }

    {inputs, updated_param_names ++ param_names, [node | nodes], op_counts, cache}
  end

  ## Convolution

  defp to_onnx(
         %Axon.Node{
           id: id,
           op: :conv,
           name: name_fn,
           parent: [inp_id],
           parameters: params,
           opts: opts
         },
         nodes_map,
         templates,
         inputs,
         param_names,
         nodes,
         op_counts,
         cache
       ) do
    {inputs, param_names, nodes, op_counts, cache} =
      to_onnx(
        nodes_map[inp_id],
        nodes_map,
        templates,
        inputs,
        param_names,
        nodes,
        op_counts,
        cache
      )

    inp_name = cache[inp_id]

    {name, op_counts, cache} =
      case cache do
        %{^id => name} ->
          {name, op_counts, cache}

        %{} ->
          name = name_fn.(:conv, op_counts)
          op_counts = Map.update(op_counts, :conv, 1, fn x -> x + 1 end)
          cache = Map.put(cache, id, name)
          {name, op_counts, cache}
      end

    input_shape = Axon.get_output_shape(%Axon{output: inp_id, nodes: nodes_map}, templates)
    strides = opts[:strides] || 1
    strides = list_or_duplicate(:strides, strides, Nx.rank(input_shape) - 2)
    padding = opts[:padding]

    strides_attr = to_attr("strides", :INTS, strides)

    padding_attr =
      case padding do
        :valid ->
          to_attr("auto_pad", :STRING, "VALID")

        :same ->
          to_attr("auto_pad", :STRING, "SAME_UPPER")

        padding when is_list(padding) ->
          {pad_begins, pad_ends} = Enum.unzip(padding)
          to_attr("pads", :INTS, pad_begins ++ pad_ends)
      end

    # TODO: Dilations

    updated_param_names =
      Enum.map(params, fn %{name: p_name} ->
        name <> "_" <> p_name
      end)

    node = %Node{
      input: [inp_name | updated_param_names],
      output: [name],
      name: name,
      attribute: [strides_attr, padding_attr],
      op_type: "Conv"
    }

    {inputs, updated_param_names ++ param_names, [node | nodes], op_counts, cache}
  end

  ## Pooling

  @supported_pooling [:max_pool, :avg_pool, :lp_pool]

  defp to_onnx(
         %Axon.Node{id: id, op: pool, name: name_fn, parent: [inp_id], opts: opts},
         nodes_map,
         templates,
         inputs,
         param_names,
         nodes,
         op_counts,
         cache
       )
       when pool in @supported_pooling do
    {inputs, param_names, nodes, op_counts, cache} =
      to_onnx(
        nodes_map[inp_id],
        nodes_map,
        templates,
        inputs,
        param_names,
        nodes,
        op_counts,
        cache
      )

    inp_name = cache[inp_id]

    {name, op_counts, cache} =
      case cache do
        %{^id => name} ->
          {name, op_counts, cache}

        %{} ->
          name = name_fn.(pool, op_counts)
          op_counts = Map.update(op_counts, pool, 1, fn x -> x + 1 end)
          cache = Map.put(cache, id, name)
          {name, op_counts, cache}
      end

    input_shape = Axon.get_output_shape(%Axon{output: inp_id, nodes: nodes_map}, templates)

    kernel_size = tuple_or_duplicate(:kernel_size, opts[:kernel_size], Nx.rank(input_shape) - 2)
    strides = opts[:strides] || Tuple.to_list(kernel_size)
    strides = list_or_duplicate(:strides, strides, Nx.rank(input_shape) - 2)
    padding = opts[:padding]

    strides_attr = to_attr("strides", :INTS, strides)
    kernel_shape_attr = to_attr("kernel_shape", :INTS, Tuple.to_list(kernel_size))

    padding_attr =
      case padding do
        :valid ->
          to_attr("auto_pad", :STRING, "VALID")

        :same ->
          to_attr("auto_pad", :STRING, "SAME_UPPER")

        padding when is_list(padding) ->
          {pad_begins, pad_ends} = Enum.unzip(padding)
          to_attr("pads", :INTS, pad_begins ++ pad_ends)
      end

    # TODO: Dilations

    {op_type, extra_attrs} =
      case pool do
        :lp_pool ->
          p_attr = to_attr("p", :INT, opts[:norm])
          {"LpPool", [p_attr]}

        :max_pool ->
          {"MaxPool", []}

        :avg_pool ->
          count_include_pad_attr = to_attr("count_include_pad", :INT, 1)
          {"AveragePool", [count_include_pad_attr]}
      end

    node_inputs = [inp_name]

    node = %Node{
      input: node_inputs,
      output: [name],
      name: name,
      attribute: [padding_attr, strides_attr, kernel_shape_attr | extra_attrs],
      op_type: op_type
    }

    {inputs, param_names, [node | nodes], op_counts, cache}
  end

  ## Global Pooling

  @supported_global_pooling [:global_avg_pool, :global_lp_pool, :global_max_pool]

  defp to_onnx(
         %Axon.Node{
           id: id,
           op: pool,
           name: name_fn,
           parent: [inp_id],
           opts: opts
         },
         nodes_map,
         templates,
         inputs,
         param_names,
         nodes,
         op_counts,
         cache
       )
       when pool in @supported_global_pooling do
    {inputs, param_names, nodes, op_counts, cache} =
      to_onnx(
        nodes_map[inp_id],
        nodes_map,
        templates,
        inputs,
        param_names,
        nodes,
        op_counts,
        cache
      )

    inp_name = cache[inp_id]

    {name, op_counts, cache} =
      case cache do
        %{^id => name} ->
          {name, op_counts, cache}

        %{} ->
          name = name_fn.(pool, op_counts)
          op_counts = Map.update(op_counts, pool, 1, fn x -> x + 1 end)
          cache = Map.put(cache, id, name)
          {name, op_counts, cache}
      end

    keep_axes = opts[:keep_axes]

    {op_type, attrs} =
      case pool do
        :global_avg_pool ->
          {"GlobalAveragePool", []}

        :global_lp_pool ->
          {"GlobalLpPool", [to_attr("p", :INT, opts[:norm])]}

        :global_max_pool ->
          {"GlobalMaxPool", []}
      end

    node_inputs = [inp_name]

    nodes =
      if keep_axes do
        node = %Node{
          input: node_inputs,
          output: [name],
          name: name,
          attribute: attrs,
          op_type: op_type
        }

        [node | nodes]
      else
        pre_squeeze_name = name <> "_pre_squeeze"

        pre_squeeze_node = %Node{
          input: node_inputs,
          output: [pre_squeeze_name],
          name: pre_squeeze_name,
          attribute: attrs,
          op_type: op_type
        }

        constant_name = name <> "_squeeze_axes"
        shape = Axon.get_output_shape(%Axon{output: inp_id, nodes: nodes_map}, templates)
        axes = Enum.to_list(2..(Nx.rank(shape) - 1)//1)
        axes_tensor = nx_to_tensor_proto(constant_name, Nx.tensor(axes))
        value_attr = to_attr("value", :TENSOR, axes_tensor)

        constant_node = %Node{
          output: [constant_name],
          name: constant_name,
          attribute: [value_attr],
          op_type: "Constant"
        }

        node = %Node{
          input: [pre_squeeze_name, constant_name],
          output: [name],
          name: name,
          op_type: "Squeeze"
        }

        [node, constant_node, pre_squeeze_node | nodes]
      end

    {inputs, param_names, nodes, op_counts, cache}
  end

  ## Activations

  @supported_activations [
    {:celu, "Celu"},
    {:elu, "Elu"},
    {:exp, "Exp"},
    {:hard_sigmoid, "HardSigmoid"},
    {:leaky_relu, "LeakyRelu"},
    {:linear, "Identity"},
    {:relu, "Relu"},
    {:sigmoid, "Sigmoid"},
    {:selu, "Selu"},
    {:softmax, "Softmax"},
    {:softplus, "Softplus"},
    {:softsign, "Softsign"},
    {:tanh, "Tanh"}
  ]

  for {op, onnx_op} <- @supported_activations do
    defp to_onnx(
           %Axon.Node{id: id, op: unquote(op), name: name_fn, parent: [inp_id]},
           nodes_map,
           templates,
           inputs,
           param_names,
           nodes,
           op_counts,
           cache
         ) do
      {inputs, param_names, nodes, op_counts, cache} =
        to_onnx(
          nodes_map[inp_id],
          nodes_map,
          templates,
          inputs,
          param_names,
          nodes,
          op_counts,
          cache
        )

      input_name = cache[inp_id]

      {name, op_counts, cache} =
        case cache do
          %{^id => name} ->
            {name, op_counts, cache}

          %{} ->
            name = name_fn.(unquote(op), op_counts)
            op_counts = Map.update(op_counts, unquote(op), 1, fn x -> x + 1 end)
            cache = Map.put(cache, id, name)
            {name, op_counts, cache}
        end

      node_inputs = [input_name]

      node = %Node{
        input: node_inputs,
        output: [name],
        name: name,
        op_type: unquote(onnx_op)
      }

      {inputs, param_names, [node | nodes], op_counts, cache}
    end
  end

  ## Stochastic

  @supported_dropout_layers [:dropout, :spatial_droput, :feature_alpha_dropout, :alpha_dropout]

  defp to_onnx(
         %Axon.Node{
           id: id,
           op: op,
           name: name_fn,
           parent: [inp_id]
         },
         nodes_map,
         templates,
         inputs,
         param_names,
         nodes,
         op_counts,
         cache
       )
       when op in @supported_dropout_layers do
    {inputs, param_names, nodes, op_counts, cache} =
      to_onnx(
        nodes_map[inp_id],
        nodes_map,
        templates,
        inputs,
        param_names,
        nodes,
        op_counts,
        cache
      )

    input_name = cache[inp_id]

    {name, op_counts, cache} =
      case cache do
        %{^id => name} ->
          {name, op_counts, cache}

        %{} ->
          name = name_fn.(op, op_counts)
          op_counts = Map.update(op_counts, op, 1, fn x -> x + 1 end)
          cache = Map.put(cache, id, name)
          {name, op_counts, cache}
      end

    # For now just forward with an identity
    node = %Node{
      input: [input_name],
      output: [name],
      name: name,
      op_type: "Identity"
    }

    # Just forward to the next layer
    {inputs, param_names, [node | nodes], op_counts, cache}
  end

  defp to_attr(name, type, value) do
    case type do
      :INT ->
        %Attribute{name: name, type: :INT, i: value}

      :INTS ->
        %Attribute{name: name, type: :INTS, ints: value}

      :STRING ->
        %Attribute{name: name, type: :STRING, s: value}

      :TENSOR ->
        %Attribute{name: name, type: :TENSOR, t: value}
    end
  end

  defp to_initializers(params_or_initializers, param_names) do
    param_names
    |> Enum.map(fn param ->
      nx_to_tensor_proto(param, params_or_initializers[param])
    end)
  end

  defp to_value_info(%Axon.Node{id: id, op: op, name: name_fn}, shape, op_counts, cache) do
    {name, op_counts, cache} =
      case cache do
        %{^id => name} ->
          {name, op_counts, cache}

        %{} ->
          name = name_fn.(op, op_counts)
          op_counts = Map.update(op_counts, op, 1, fn x -> x + 1 end)
          cache = Map.put(cache, id, name)
          {name, op_counts, cache}
      end

    input_type = %Type{value: {:tensor_type, to_placeholder(shape)}}
    {%Value{name: name, type: input_type}, op_counts, cache}
  end

  defp to_value_info(param_name, shape) do
    input_type = %Type{value: {:tensor_type, to_placeholder(shape)}}
    %Value{name: param_name, type: input_type}
  end

  defp to_placeholder(shape) do
    %Placeholder{shape: to_tensor_shape_proto(shape), elem_type: 1}
  end

  defp to_tensor_proto(tensor) do
    dims = Tuple.to_list(Nx.shape(tensor))
    type = nx_type_to_onnx_type(Nx.type(tensor))
    data = Nx.to_binary(tensor)

    %Tensor{dims: dims, data_type: type, raw_data: data}
  end

  defp to_tensor_shape_proto(shape) do
    dims =
      shape
      |> Tuple.to_list()
      |> Enum.map(fn
        nil ->
          %Dimension{value: {:dim_param, 1}}

        value ->
          %Dimension{value: {:dim_value, value}}
      end)

    %Shape{dim: dims}
  end

  defp nx_to_tensor_proto(param_name, tensor) do
    dims = Nx.shape(tensor) |> Tuple.to_list()
    # TODO: fix
    data_type =
      case Nx.type(tensor) do
        {:f, 32} ->
          1

        {:s, 64} ->
          7
      end

    raw_data = Nx.to_binary(tensor)
    %Onnx.TensorProto{name: param_name, dims: dims, data_type: data_type, raw_data: raw_data}
  end

  defp tuple_or_duplicate(key, tuple_or_integer, rank) do
    cond do
      is_tuple(tuple_or_integer) ->
        if tuple_size(tuple_or_integer) != rank do
          raise ArgumentError,
                "expected #{inspect(key)} to be a #{rank}-element tuple, " <>
                  "got: #{inspect(tuple_or_integer)}"
        end

        tuple_or_integer

      is_integer(tuple_or_integer) ->
        Tuple.duplicate(tuple_or_integer, rank)

      true ->
        raise ArgumentError,
              "expected #{inspect(key)} to be an integer or a tuple, " <>
                "got: #{inspect(tuple_or_integer)}"
    end
  end

  defp list_or_duplicate(key, list_or_integer, rank) do
    cond do
      is_list(list_or_integer) ->
        if length(list_or_integer) != rank do
          raise ArgumentError,
                "expected #{inspect(key)} to be a #{rank}-element list, " <>
                  "got: #{inspect(list_or_integer)}"
        end

        list_or_integer

      is_integer(list_or_integer) ->
        List.duplicate(list_or_integer, rank)

      true ->
        raise ArgumentError,
              "expected #{inspect(key)} to be an integer or a list, " <>
                "got: #{inspect(list_or_integer)}"
    end
  end
end
