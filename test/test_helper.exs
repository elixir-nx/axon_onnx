# TODO: Add torchvision back to CI, for now the models
# are too large to test and generate all at once in an
# action
# TODO: Add transformers back to CI, for now the models
# are too large and the upstream dependency is too fragile
# to include in an action
ExUnit.start(exclude: [:torchvision, :transformers])

defmodule OnnxTestHelper do
  @moduledoc """
  Helpers for running ONNX's suite of tests on imported models.
  """
  require Axon
  require Logger

  @cache_dir Path.join([File.cwd!(), ".test-cache"])

  @doc """
  Serializes and tests model against test cases.
  This function will generate N cases and serialize them
  along with the model, storing them in the test cache. It invokes
  The script `check_onnx_model.py` to ensure ONNX runtime results
  are consistent with Axon results.
  """
  def serialize_and_test_model!(%Axon{name: name} = axon_model, input_shape, opts \\ []) do
    num_cases = opts[:num_tests] || 5
    model_name = opts[:name] || name
    cache_dir = Path.join([@cache_dir, model_name])
    File.mkdir_p!(cache_dir)

    model_path = Path.join([cache_dir, "#{model_name}.onnx"])

    params = Axon.init(axon_model, Nx.template(input_shape, {:f, 32}), %{}, compiler: EXLA)

    Enum.each(1..num_cases//1, fn n ->
      test_path = Path.join([cache_dir, "test_data_set_#{n}"])
      File.mkdir_p!(test_path)

      inp = %{"input" => Nx.random_uniform(input_shape, type: {:f, 32})}

      out = Axon.predict(axon_model, params, inp)

      inp
      |> Map.values()
      |> Enum.with_index(fn inp, i ->
        nx_to_tensor_proto(inp, Path.join([test_path, "input_#{i}.pb"]))
      end)

      nx_to_tensor_proto(out, Path.join([test_path, "output_0.pb"]))
    end)

    AxonOnnx.export(axon_model, %{"input" => Nx.template(input_shape, {:f, 32})}, params,
      path: model_path
    )

    # Run check script
    {_, exit_code} =
      System.cmd("python3", ["scripts/check_onnx_model.py", model_path], into: IO.stream())

    unless exit_code == 0 do
      raise "Model serialization failed for #{model_name}"
    end
  end

  # TODO: Maybe this should be in utils
  defp nx_to_tensor_proto(tensor, path) do
    dims = Nx.shape(tensor) |> Tuple.to_list()
    # TODO: fix
    data_type = 1
    raw_data = Nx.to_binary(tensor)
    tp = %Onnx.TensorProto{dims: dims, data_type: data_type, raw_data: raw_data}

    encoded_tp = Onnx.TensorProto.encode!(tp)
    {:ok, file} = File.open(path, [:write])
    IO.binwrite(file, encoded_tp)
    File.close(file)
  end

  @doc """
  Tests model agains given ONNX test case.
  """
  def check_onnx_test_case!(type, test_name, opts \\ []) do
    test_path = Path.join(["test", "cases", type, test_name])
    model_path = Path.join([test_path, "model.onnx"])
    data_paths = Path.wildcard(Path.join([test_path, "test_data_set_*"]))

    {model, params} = AxonOnnx.import(model_path)

    inputs = Axon.get_inputs(model)

    data_paths
    |> Enum.map(fn data_path ->
      input_paths = Path.wildcard(Path.join([data_path, "input_*.pb"]))
      output_paths = Path.wildcard(Path.join([data_path, "output_*.pb"]))

      inp_tensors =
        input_paths
        |> Enum.map(&pb_to_tensor/1)
        |> Enum.zip(Map.keys(inputs))
        |> Map.new(fn {v, k} -> {k, v} end)

      out_tensors = Enum.map(output_paths, &pb_to_tensor/1)

      actual_outputs = Axon.predict(model, params, inp_tensors, opts)

      case out_tensors do
        [expected_output] ->
          assert_all_close!(actual_outputs, expected_output)

        [_ | _] = expected_outputs ->
          Enum.zip_with(Tuple.to_list(actual_outputs), expected_outputs, &assert_all_close!/2)
      end
    end)
  end

  @doc """
  Tests given ONNX model.
  """
  def check_onnx_model!(model_name, opts \\ []) do
    test_name = "test_" <> model_name
    base_path = Path.join(["test", "cases", "real", test_name])
    test_path = Path.join([base_path, "data.json"])
    model_path = Path.join([base_path, "model.onnx"])

    if File.exists?(model_path) do
      check_onnx_test_case!("real", test_name, opts)
    else
      data =
        test_path
        |> File.read!()
        |> Jason.decode!()

      Logger.info("Downloading #{model_name} from #{data["url"]}")

      {:ok, files} =
        :erl_tar.extract({:binary, Req.get!(data["url"]).body}, [:compressed, :memory])

      files
      |> Enum.map(fn {fname, data} ->
        [_ | rest] = Path.split(fname)
        path = Path.join([base_path | rest])

        if File.exists?(path) do
          :ok
        else
          File.mkdir_p!(Path.dirname(path))
          File.write!(path, data)
        end
      end)

      check_onnx_test_case!("real", test_name, opts)
    end
  end

  @doc """
  Checks the given transformer model is imported.
  """
  def check_onnx_transformer!(model_name, opts \\ []) do
    base_path = Path.join(["test", "cases", "transformers"])
    File.mkdir_p(base_path)

    path = Path.join([base_path, model_name])
    model_path = Path.join([path, "model.onnx"])

    unless File.exists?(model_path) do
      System.cmd("python3", ["-m", "transformers.onnx", "--model=#{model_name}", "#{path}"])
    end

    # Ensure import
    AxonOnnx.import(model_path, opts)
  end

  defp assert_all_close!(x, y) do
    # Normalize outputs
    res = Nx.all_close(x, y, atol: 1.0e-3) |> Nx.to_number()

    unless res == 1 do
      raise "expected #{inspect(x)} to be within tolerance of #{inspect(y)}"
    end

    :ok
  end

  # Parses the protobuf file into an Nx tensor.
  def pb_to_tensor(pb_path) do
    pb_path
    |> File.read!()
    |> Onnx.TensorProto.decode!()
    |> tensor!()
  end

  defp tensor!(%Onnx.TensorProto{data_type: dtype, dims: dims} = tensor) do
    shape = List.to_tuple(dims)

    case dtype do
      1 ->
        to_nx_tensor(tensor.float_data, tensor.raw_data, {:f, 32}, shape)

      2 ->
        to_nx_tensor(tensor.int32_data, tensor.raw_data, {:u, 8}, shape)

      3 ->
        to_nx_tensor(tensor.int32_data, tensor.raw_data, {:s, 8}, shape)

      4 ->
        to_nx_tensor(tensor.int32_data, tensor.raw_data, {:u, 16}, shape)

      5 ->
        to_nx_tensor(tensor.int32_data, tensor.raw_data, {:s, 16}, shape)

      6 ->
        to_nx_tensor(tensor.int32_data, tensor.raw_data, {:s, 32}, shape)

      7 ->
        to_nx_tensor(tensor.int64_data, tensor.raw_data, {:s, 64}, shape)

      8 ->
        raise "unsupported Nx tensor type: string"

      9 ->
        to_nx_tensor(tensor.int32_data, tensor.raw_data, {:u, 8}, shape)

      10 ->
        to_nx_tensor(tensor.int32_data, tensor.raw_data, {:f, 16}, shape)

      11 ->
        to_nx_tensor(tensor.double_data, tensor.raw_data, {:f, 64}, shape)

      12 ->
        to_nx_tensor(tensor.uint64_data, tensor.raw_data, {:u, 32}, shape)

      13 ->
        to_nx_tensor(tensor.uint64_data, tensor.raw_data, {:u, 64}, shape)

      14 ->
        # TODO(seanmor5): When complex is supported, tensor.float_data
        raise "unsupported Nx tensor type: C64"

      15 ->
        # TODO(seanmor5): When complex is supported, tensor.double_data
        raise "unsupported Nx tensor type: C128"

      16 ->
        to_nx_tensor([], tensor.raw_data, {:bf, 16}, shape)
    end
  end

  defp to_nx_tensor([], <<>>, _, _) do
    raise "unsupported empty Nx tensor"
  end

  defp to_nx_tensor([], raw, type, shape) do
    raw
    |> Nx.from_binary(type)
    |> Nx.reshape(shape)
  end

  defp to_nx_tensor(data, _, type, shape) do
    data
    |> Nx.tensor(type: type)
    |> Nx.reshape(shape)
  end
end

require Logger

cases_path = Path.join([__DIR__], "cases")
Logger.info("Generating ONNX test cases...")

if not File.exists?(cases_path) do
  # Generate cases
  # requires onnx python package: pip3 install onnx
  System.cmd("backend-test-tools", ["generate-data"])

  # Get cases path
  {path, _} =
    System.cmd("python3", [
      "-c",
      "from onnx.backend import test; import os; print(os.path.dirname(test.__file__), end='', sep='')"
    ])

  path = Path.join([path, "data"])

  # Move all to test directory
  File.mkdir_p!(cases_path)

  path
  |> File.ls!()
  |> Enum.each(fn base_path ->
    src = Path.join([path, base_path])
    dst = Path.join([cases_path, base_path])
    File.cp_r!(src, dst)
  end)
end

Logger.info("Finished generating test cases")
