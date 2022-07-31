defmodule DeserializeTest do
  use ExUnit.Case, async: false
  import OnnxTestHelper

  describe "node tests" do
    test "Abs" do
      check_onnx_test_case!("node", "test_abs")
    end

    test "Acos" do
      check_onnx_test_case!("node", "test_acos")
      check_onnx_test_case!("node", "test_acos_example")
    end

    test "Acosh" do
      check_onnx_test_case!("node", "test_acosh")
      check_onnx_test_case!("node", "test_acosh_example")
    end

    test "Add" do
      check_onnx_test_case!("node", "test_add")
      check_onnx_test_case!("node", "test_add_bcast")
      check_onnx_test_case!("node", "test_add_uint8")
    end

    test "And" do
      check_onnx_test_case!("node", "test_and2d")
      check_onnx_test_case!("node", "test_and3d")
      check_onnx_test_case!("node", "test_and4d")
      check_onnx_test_case!("node", "test_and_bcast3v1d")
      check_onnx_test_case!("node", "test_and_bcast3v2d")
      check_onnx_test_case!("node", "test_and_bcast4v2d")
      check_onnx_test_case!("node", "test_and_bcast4v3d")
      check_onnx_test_case!("node", "test_and_bcast4v4d")
    end

    test "ArgMax" do
      check_onnx_test_case!("node", "test_argmax_default_axis_example")
      check_onnx_test_case!("node", "test_argmax_default_axis_example_select_last_index")
      check_onnx_test_case!("node", "test_argmax_default_axis_random")
      check_onnx_test_case!("node", "test_argmax_default_axis_random_select_last_index")
      check_onnx_test_case!("node", "test_argmax_keepdims_example")
      check_onnx_test_case!("node", "test_argmax_keepdims_example_select_last_index")
      check_onnx_test_case!("node", "test_argmax_keepdims_random")
      check_onnx_test_case!("node", "test_argmax_keepdims_random_select_last_index")
      check_onnx_test_case!("node", "test_argmax_negative_axis_keepdims_example")

      check_onnx_test_case!(
        "node",
        "test_argmax_negative_axis_keepdims_example_select_last_index"
      )

      check_onnx_test_case!("node", "test_argmax_negative_axis_keepdims_random")
      check_onnx_test_case!("node", "test_argmax_negative_axis_keepdims_random_select_last_index")
      check_onnx_test_case!("node", "test_argmax_no_keepdims_example")
      check_onnx_test_case!("node", "test_argmax_no_keepdims_example_select_last_index")
      check_onnx_test_case!("node", "test_argmax_no_keepdims_random")
      check_onnx_test_case!("node", "test_argmax_no_keepdims_random_select_last_index")
    end

    test "ArgMin" do
      check_onnx_test_case!("node", "test_argmin_default_axis_example")
      check_onnx_test_case!("node", "test_argmin_default_axis_example_select_last_index")
      check_onnx_test_case!("node", "test_argmin_default_axis_random")
      check_onnx_test_case!("node", "test_argmin_default_axis_random_select_last_index")
      check_onnx_test_case!("node", "test_argmin_keepdims_example")
      check_onnx_test_case!("node", "test_argmin_keepdims_example_select_last_index")
      check_onnx_test_case!("node", "test_argmin_keepdims_random")
      check_onnx_test_case!("node", "test_argmin_keepdims_random_select_last_index")
      check_onnx_test_case!("node", "test_argmin_negative_axis_keepdims_example")

      check_onnx_test_case!(
        "node",
        "test_argmin_negative_axis_keepdims_example_select_last_index"
      )

      check_onnx_test_case!("node", "test_argmin_negative_axis_keepdims_random")
      check_onnx_test_case!("node", "test_argmin_negative_axis_keepdims_random_select_last_index")
      check_onnx_test_case!("node", "test_argmin_no_keepdims_example")
      check_onnx_test_case!("node", "test_argmin_no_keepdims_example_select_last_index")
      check_onnx_test_case!("node", "test_argmin_no_keepdims_random")
      check_onnx_test_case!("node", "test_argmin_no_keepdims_random_select_last_index")
    end

    test "Asin" do
      check_onnx_test_case!("node", "test_asin")
      check_onnx_test_case!("node", "test_asin_example")
    end

    test "Asinh" do
      check_onnx_test_case!("node", "test_asinh")
      check_onnx_test_case!("node", "test_asinh_example")
    end

    test "AveragePool" do
      check_onnx_test_case!("node", "test_averagepool_1d_default")
      check_onnx_test_case!("node", "test_averagepool_2d_default")
      # TODO: Count include pad
      # check_onnx_test_case!("node", "test_averagepool_2d_pads")
      check_onnx_test_case!("node", "test_averagepool_2d_pads_count_include_pad")
      # check_onnx_test_case!("node", "test_averagepool_2d_precomputed_pads")
      check_onnx_test_case!("node", "test_averagepool_2d_precomputed_pads_count_include_pad")
      # check_onnx_test_case!("node", "test_averagepool_2d_precomputed_same_upper")
      check_onnx_test_case!("node", "test_averagepool_2d_precomputed_strides")
      # check_onnx_test_case!("node", "test_averagepool_2d_same_lower")
      # check_onnx_test_case!("node", "test_averagepool_2d_same_upper")
      check_onnx_test_case!("node", "test_averagepool_2d_strides")
      check_onnx_test_case!("node", "test_averagepool_3d_default")
      # TODO: Ceil mode
      # check_onnx_test_case!("node", "test_averagepool_2d_ceil")
    end

    # test "BatchNormalization" do
    #   check_onnx_test_case!("node", "test_batchnorm_epsilon")
    #   check_onnx_test_case!("node", "test_batchnorm_epsilon_training_mode")
    #   check_onnx_test_case!("node", "test_batchnorm_example")
    #   check_onnx_test_case!("node", "test_batchnorm_example_training_mode")
    # end

    test "BitShift" do
      check_onnx_test_case!("node", "test_bitshift_left_uint8")
      check_onnx_test_case!("node", "test_bitshift_left_uint16")
      check_onnx_test_case!("node", "test_bitshift_left_uint32")
      check_onnx_test_case!("node", "test_bitshift_left_uint64")
      check_onnx_test_case!("node", "test_bitshift_right_uint8")
      check_onnx_test_case!("node", "test_bitshift_right_uint16")
      check_onnx_test_case!("node", "test_bitshift_right_uint32")
      check_onnx_test_case!("node", "test_bitshift_right_uint64")
    end

    test "Cast" do
      # check_onnx_test_case!("node", "test_cast_BFLOAT16_to_FLOAT")
      check_onnx_test_case!("node", "test_cast_DOUBLE_to_FLOAT")
      check_onnx_test_case!("node", "test_cast_DOUBLE_to_FLOAT16")
      check_onnx_test_case!("node", "test_cast_FLOAT16_to_FLOAT")
      # check_onnx_test_case!("node", "test_cast_FLOAT_to_BFLOAT16")
      check_onnx_test_case!("node", "test_cast_FLOAT_to_DOUBLE")
      check_onnx_test_case!("node", "test_cast_FLOAT_to_FLOAT16")
      # check_onnx_test_case!("node", "test_cast_FLOAT_to_STRING")
      # check_onnx_test_case!("node", "test_cast_STRING_to_FLOAT")
    end

    test "Ceil" do
      check_onnx_test_case!("node", "test_ceil")
      check_onnx_test_case!("node", "test_ceil_example")
    end

    test "Celu" do
      check_onnx_test_case!("node", "test_celu")
      check_onnx_test_case!("node", "test_celu_expanded")
    end

    # test "Clip" do
    #   check_onnx_test_case!("node", "test_clip")
    #   # check_onnx_test_case!("node", "test_clip_default_int8_max")
    #   # check_onnx_test_case!("node", "test_clip_default_min")
    #   check_onnx_test_case!("node", "test_clip_outbounds")
    #   # check_onnx_test_case!("node", "test_clip_default_inbounds")
    #   # check_onnx_test_case!("node", "test_clip_default_int8_min")
    #   check_onnx_test_case!("node", "test_clip_example")
    #   check_onnx_test_case!("node", "test_clip_splitbounds")
    #   # check_onnx_test_case!("node", "test_clip_default_int8_inbounds")
    #   # check_onnx_test_case!("node", "test_clip_default_max")
    #   check_onnx_test_case!("node", "test_clip_inbounds")
    # end

    test "Concat" do
      check_onnx_test_case!("node", "test_concat_1d_axis_0")
      check_onnx_test_case!("node", "test_concat_1d_axis_negative_1")
      check_onnx_test_case!("node", "test_concat_2d_axis_0")
      check_onnx_test_case!("node", "test_concat_2d_axis_1")
      check_onnx_test_case!("node", "test_concat_2d_axis_negative_1")
      check_onnx_test_case!("node", "test_concat_2d_axis_negative_2")
      check_onnx_test_case!("node", "test_concat_3d_axis_0")
      check_onnx_test_case!("node", "test_concat_3d_axis_1")
      check_onnx_test_case!("node", "test_concat_3d_axis_2")
      check_onnx_test_case!("node", "test_concat_3d_axis_negative_1")
      check_onnx_test_case!("node", "test_concat_3d_axis_negative_2")
      check_onnx_test_case!("node", "test_concat_3d_axis_negative_3")
    end

    test "Constant" do
      check_onnx_test_case!("node", "test_constant")
      # check_onnx_test_case!("node", "test_constant_pad")
    end

    # test "ConstantOfShape" do
    #   check_onnx_test_case!("node", "test_constantofshape_float_ones")
    #   check_onnx_test_case!("node", "test_constantofshape_int_shape_zero")
    #   check_onnx_test_case!("node", "test_constantofshape_int_zeros")
    # end

    # test "Conv" do
    #   check_onnx_test_case!("node", "test_conv_with_autopad_same")
    #   check_onnx_test_case!("node", "test_conv_with_strides_and_asymmetric_padding")
    #   check_onnx_test_case!("node", "test_conv_with_strides_no_padding")
    #   check_onnx_test_case!("node", "test_conv_with_strides_padding")
    # end

    # test "ConvInteger" do
    #   check_onnx_test_case!("node", "test_convinteger_with_padding")
    #   check_onnx_test_case!("node", "test_convinteger_without_padding")
    # end

    # test "ConvTranspose" do
    #   check_onnx_test_case!("node", "test_convtranspose")
    #   check_onnx_test_case!("node", "test_convtranspose_1d")
    #   check_onnx_test_case!("node", "test_convtranspose_3d")
    #   check_onnx_test_case!("node", "test_convtranspose_autopad_same")
    #   check_onnx_test_case!("node", "test_convtranspose_dilations")
    #   check_onnx_test_case!("node", "test_convtranspose_kernel_shape")
    #   check_onnx_test_case!("node", "test_convtranspose_output_shape")
    #   check_onnx_test_case!("node", "test_convtranspose_pad")
    #   check_onnx_test_case!("node", "test_convtranspose_pads")
    #   check_onnx_test_case!("node", "test_convtranspose_with_kernel")
    # end

    test "Cos" do
      check_onnx_test_case!("node", "test_cos")
      check_onnx_test_case!("node", "test_cos_example")
    end

    test "Cosh" do
      check_onnx_test_case!("node", "test_cosh")
      check_onnx_test_case!("node", "test_cosh_example")
    end

    # test "CumSum" do
    #   check_onnx_test_case!("node", "test_cumsum_1d")
    #   check_onnx_test_case!("node", "test_cumsum_1d_exclusive")
    #   check_onnx_test_case!("node", "test_cumsum_1d_reverse")
    #   check_onnx_test_case!("node", "test_cumsum_1d_reverse_exclusive")
    #   check_onnx_test_case!("node", "test_cumsum_2d_axis_0")
    #   check_onnx_test_case!("node", "test_cumsum_2d_axis_1")
    #   check_onnx_test_case!("node", "test_cumsum_2d_negative_axis")
    # end

    test "Div" do
      check_onnx_test_case!("node", "test_div")
      check_onnx_test_case!("node", "test_div_bcast")
      check_onnx_test_case!("node", "test_div_example")
      # TODO: Cast?
      # check_onnx_test_case!("node", "test_div_uint8")
    end

    test "Equal" do
      check_onnx_test_case!("node", "test_equal")
      check_onnx_test_case!("node", "test_equal_bcast")
    end

    test "Elu" do
      check_onnx_test_case!("node", "test_elu")
      check_onnx_test_case!("node", "test_elu_default")
      check_onnx_test_case!("node", "test_elu_example")
    end

    test "Erf" do
      check_onnx_test_case!("node", "test_erf")
    end

    test "Exp" do
      check_onnx_test_case!("node", "test_exp")
      check_onnx_test_case!("node", "test_exp_example")
    end

    # test "Expand" do
    #   check_onnx_test_case!("node", "test_expand_dim_changed")
    #   check_onnx_test_case!("node", "test_expand_dim_unchanged")
    # end

    # test "EyeLike" do
    #   check_onnx_test_case!("node", "test_eyelike_with_dtype")
    #   check_onnx_test_case!("node", "test_eyelike_without_dtype")
    #   check_onnx_test_case!("node", "test_eyelike_populate_off_main_diagonal")
    # end

    # test "Flatten" do
    # check_onnx_test_case!("node", "test_flatten_axis0")
    # check_onnx_test_case!("node", "test_flatten_axis1")
    # check_onnx_test_case!("node", "test_flatten_axis2")
    # check_onnx_test_case!("node", "test_flatten_axis3")
    # check_onnx_test_case!("node", "test_flatten_default_axis")
    # check_onnx_test_case!("node", "test_flatten_negative_axis1")
    # check_onnx_test_case!("node", "test_flatten_negative_axis2")
    # check_onnx_test_case!("node", "test_flatten_negative_axis3")
    # check_onnx_test_case!("node", "test_flatten_negative_axis4")
    # end

    test "Floor" do
      check_onnx_test_case!("node", "test_floor")
      check_onnx_test_case!("node", "test_floor_example")
    end

    test "Gather" do
      check_onnx_test_case!("node", "test_gather_0")
      check_onnx_test_case!("node", "test_gather_1")
      check_onnx_test_case!("node", "test_gather_2d_indices")
    end

    test "Gemm" do
      check_onnx_test_case!("node", "test_gemm_all_attributes")
      check_onnx_test_case!("node", "test_gemm_alpha")
      check_onnx_test_case!("node", "test_gemm_beta")
      check_onnx_test_case!("node", "test_gemm_default_matrix_bias")
      check_onnx_test_case!("node", "test_gemm_default_no_bias")
      check_onnx_test_case!("node", "test_gemm_default_scalar_bias")
      check_onnx_test_case!("node", "test_gemm_default_single_elem_vector_bias")
      check_onnx_test_case!("node", "test_gemm_default_vector_bias")
      check_onnx_test_case!("node", "test_gemm_default_zero_bias")
      check_onnx_test_case!("node", "test_gemm_transposeA")
      check_onnx_test_case!("node", "test_gemm_transposeB")
    end

    test "GlobalAveragePool" do
      check_onnx_test_case!("node", "test_globalaveragepool")
      check_onnx_test_case!("node", "test_globalaveragepool_precomputed")
    end

    # No tests?
    # test "GlobalLpPool" do
    # end

    test "GlobalMaxPool" do
      check_onnx_test_case!("node", "test_globalmaxpool")
      check_onnx_test_case!("node", "test_globalmaxpool_precomputed")
    end

    test "Greater" do
      check_onnx_test_case!("node", "test_greater")
      check_onnx_test_case!("node", "test_greater_bcast")
    end

    test "GreaterOrEqual" do
      check_onnx_test_case!("node", "test_greater_equal")
      check_onnx_test_case!("node", "test_greater_equal_expanded")
      check_onnx_test_case!("node", "test_greater_equal_bcast")
      check_onnx_test_case!("node", "test_greater_equal_bcast_expanded")
    end

    test "HardSigmoid" do
      check_onnx_test_case!("node", "test_hardsigmoid")
      check_onnx_test_case!("node", "test_hardsigmoid_example")
      check_onnx_test_case!("node", "test_hardsigmoid_default")
    end

    test "HardSwish" do
      check_onnx_test_case!("node", "test_hardswish")
      check_onnx_test_case!("node", "test_hardswish_expanded")
    end

    test "Identity" do
      check_onnx_test_case!("node", "test_identity")
      # TODO: Sequence types
      # check_onnx_test_case!("node", "test_identity_sequence")
    end

    # test "If" do
    #   check_onnx_test_case!("node", "test_if")
    #   check_onnx_test_case!("node", "test_if_seq")
    # end

    # test "InstanceNormalization" do
    #   check_onnx_test_case!("node", "test_instancenorm_epsilon")
    #   check_onnx_test_case!("node", "test_instancenorm_example")
    # end

    test "IsInf" do
      check_onnx_test_case!("node", "test_isinf")
      # check_onnx_test_case!("node", "test_isinf_negative")
      # check_onnx_test_case!("node", "test_isinf_positive")
    end

    test "IsNaN" do
      check_onnx_test_case!("node", "test_isnan")
    end

    test "LeakyRelu" do
      check_onnx_test_case!("node", "test_leakyrelu")
      check_onnx_test_case!("node", "test_leakyrelu_default")
      check_onnx_test_case!("node", "test_leakyrelu_example")
    end

    test "Less" do
      check_onnx_test_case!("node", "test_less")
      check_onnx_test_case!("node", "test_less_bcast")
    end

    test "LessOrEqual" do
      check_onnx_test_case!("node", "test_less_equal")
      check_onnx_test_case!("node", "test_less_equal_expanded")
      check_onnx_test_case!("node", "test_less_equal_bcast")
      check_onnx_test_case!("node", "test_less_equal_bcast_expanded")
    end

    test "Log" do
      check_onnx_test_case!("node", "test_log")
      check_onnx_test_case!("node", "test_log_example")
    end

    test "LogSoftmax" do
      check_onnx_test_case!("node", "test_logsoftmax_axis_0")
      # TODO: ReduceMax
      # check_onnx_test_case!("node", "test_logsoftmax_axis_0_expanded")
      check_onnx_test_case!("node", "test_logsoftmax_axis_1")
      # TODO: ReduceMax
      # check_onnx_test_case!("node", "test_logsoftmax_axis_1_expanded")
      check_onnx_test_case!("node", "test_logsoftmax_axis_2")
      # TODO: ReduceMax
      # check_onnx_test_case!("node", "test_logsoftmax_axis_2_expanded")
      check_onnx_test_case!("node", "test_logsoftmax_default_axis")
      # TODO: ReduceMax
      # check_onnx_test_case!("node", "test_logsoftmax_default_axis_expanded")
      check_onnx_test_case!("node", "test_logsoftmax_large_number")
      # TODO: ReduceMax
      # check_onnx_test_case!("node", "test_logsoftmax_large_number_expanded")
      check_onnx_test_case!("node", "test_logsoftmax_negative_axis")
      # TODO: ReduceMax
      # check_onnx_test_case!("node", "test_logsoftmax_negative_axis_expanded")
      check_onnx_test_case!("node", "test_logsoftmax_example_1")
      # TODO: ReduceMax
      # check_onnx_test_case!("node", "test_logsoftmax_example_1_expanded")
    end

    test "LRN" do
      check_onnx_test_case!("node", "test_lrn")
      check_onnx_test_case!("node", "test_lrn_default")
    end

    test "MatMul" do
      check_onnx_test_case!("node", "test_matmul_2d")
      check_onnx_test_case!("node", "test_matmul_3d")
      check_onnx_test_case!("node", "test_matmul_4d")
    end

    test "Max" do
      check_onnx_test_case!("node", "test_max_example")
      check_onnx_test_case!("node", "test_max_float16")
      check_onnx_test_case!("node", "test_max_float32")
      check_onnx_test_case!("node", "test_max_float64")
      check_onnx_test_case!("node", "test_max_int8")
      check_onnx_test_case!("node", "test_max_int16")
      check_onnx_test_case!("node", "test_max_int32")
      check_onnx_test_case!("node", "test_max_int64")
      check_onnx_test_case!("node", "test_max_uint8")
      check_onnx_test_case!("node", "test_max_uint16")
      check_onnx_test_case!("node", "test_max_uint32")
      check_onnx_test_case!("node", "test_max_uint64")
      check_onnx_test_case!("node", "test_max_one_input")
      check_onnx_test_case!("node", "test_max_two_inputs")
    end

    test "MaxPool" do
      check_onnx_test_case!("node", "test_maxpool_1d_default")
      check_onnx_test_case!("node", "test_maxpool_2d_default")
      # TODO: https://github.com/elixir-nx/axon/issues/185
      # check_onnx_test_case!("node", "test_maxpool_2d_dilations")
      check_onnx_test_case!("node", "test_maxpool_2d_pads")
      # TODO: Adjust pads for same lower behavior
      # check_onnx_test_case!("node", "test_maxpool_2d_same_lower")
      check_onnx_test_case!("node", "test_maxpool_2d_same_upper")
      check_onnx_test_case!("node", "test_maxpool_2d_strides")
      check_onnx_test_case!("node", "test_maxpool_2d_uint8")
      check_onnx_test_case!("node", "test_maxpool_2d_precomputed_pads")
      check_onnx_test_case!("node", "test_maxpool_2d_precomputed_same_upper")
      check_onnx_test_case!("node", "test_maxpool_2d_precomputed_strides")
      # TODO: Return indices in MaxPool
      # check_onnx_test_case!("node", "test_maxpool_with_argmax_2d_precomputed_pads")
      # check_onnx_test_case!("node", "test_maxpool_with_argmax_2d_precomputed_strides")
      check_onnx_test_case!("node", "test_maxpool_3d_default")

      # TODO: Reevaluate this behavior
      assert_raise ArgumentError, ~r/invalid ceil_mode/, fn ->
        check_onnx_test_case!("node", "test_maxpool_2d_ceil")
      end
    end

    # test "Mean" do
    #   check_onnx_test_case!("node", "test_mean_example")
    #   check_onnx_test_case!("node", "test_mean_one_input")
    #   check_onnx_test_case!("node", "test_mean_two_inputs")
    # end

    test "Min" do
      check_onnx_test_case!("node", "test_min_example")
      check_onnx_test_case!("node", "test_min_float16")
      check_onnx_test_case!("node", "test_min_float32")
      check_onnx_test_case!("node", "test_min_float64")
      check_onnx_test_case!("node", "test_min_int8")
      check_onnx_test_case!("node", "test_min_int16")
      check_onnx_test_case!("node", "test_min_int32")
      check_onnx_test_case!("node", "test_min_int64")
      check_onnx_test_case!("node", "test_min_uint8")
      check_onnx_test_case!("node", "test_min_uint16")
      check_onnx_test_case!("node", "test_min_uint32")
      check_onnx_test_case!("node", "test_min_uint64")
      check_onnx_test_case!("node", "test_min_one_input")
      check_onnx_test_case!("node", "test_min_two_inputs")
    end

    test "Mod" do
      check_onnx_test_case!("node", "test_mod_broadcast")
      check_onnx_test_case!("node", "test_mod_int64_fmod")
      check_onnx_test_case!("node", "test_mod_mixed_sign_float16")
      check_onnx_test_case!("node", "test_mod_mixed_sign_float32")
      check_onnx_test_case!("node", "test_mod_mixed_sign_float64")
      # TODO: Somethings wrong here...
      # check_onnx_test_case!("node", "test_mod_mixed_sign_int8")
      # check_onnx_test_case!("node", "test_mod_mixed_sign_int16")
      # check_onnx_test_case!("node", "test_mod_mixed_sign_int32")
      # check_onnx_test_case!("node", "test_mod_mixed_sign_int64")
      check_onnx_test_case!("node", "test_mod_uint8")
      check_onnx_test_case!("node", "test_mod_uint16")
      check_onnx_test_case!("node", "test_mod_uint32")
      check_onnx_test_case!("node", "test_mod_uint64")
    end

    test "Mul" do
      check_onnx_test_case!("node", "test_mul")
      check_onnx_test_case!("node", "test_mul_bcast")
      check_onnx_test_case!("node", "test_mul_example")
      check_onnx_test_case!("node", "test_mul_uint8")
    end

    test "Neg" do
      check_onnx_test_case!("node", "test_neg")
      check_onnx_test_case!("node", "test_neg_example")
    end

    test "Not" do
      check_onnx_test_case!("node", "test_not_2d")
      check_onnx_test_case!("node", "test_not_3d")
      check_onnx_test_case!("node", "test_not_4d")
    end

    test "Or" do
      check_onnx_test_case!("node", "test_or2d")
      check_onnx_test_case!("node", "test_or3d")
      check_onnx_test_case!("node", "test_or4d")
      check_onnx_test_case!("node", "test_or_bcast3v1d")
      check_onnx_test_case!("node", "test_or_bcast3v2d")
      check_onnx_test_case!("node", "test_or_bcast4v2d")
      check_onnx_test_case!("node", "test_or_bcast4v3d")
      check_onnx_test_case!("node", "test_or_bcast4v4d")
    end

    # No tests?
    # test "Pad" do
    # end

    test "Pow" do
      check_onnx_test_case!("node", "test_pow")
      check_onnx_test_case!("node", "test_pow_bcast_array")
      check_onnx_test_case!("node", "test_pow_bcast_scalar")
      check_onnx_test_case!("node", "test_pow_example")
      check_onnx_test_case!("node", "test_pow_types_float")
      check_onnx_test_case!("node", "test_pow_types_float32_int32")
      check_onnx_test_case!("node", "test_pow_types_float32_int64")
      check_onnx_test_case!("node", "test_pow_types_float32_uint32")
      check_onnx_test_case!("node", "test_pow_types_float32_uint64")
      check_onnx_test_case!("node", "test_pow_types_int")
      check_onnx_test_case!("node", "test_pow_types_int32_float32")
      check_onnx_test_case!("node", "test_pow_types_int32_int32")
      check_onnx_test_case!("node", "test_pow_types_int64_float32")
      check_onnx_test_case!("node", "test_pow_types_int64_int64")
    end

    test "Reciprocal" do
      check_onnx_test_case!("node", "test_reciprocal")
      check_onnx_test_case!("node", "test_reciprocal_example")
    end

    test "ReduceLogSum" do
      check_onnx_test_case!("node", "test_reduce_log_sum")
      check_onnx_test_case!("node", "test_reduce_log_sum_asc_axes")
      check_onnx_test_case!("node", "test_reduce_log_sum_default")
      check_onnx_test_case!("node", "test_reduce_log_sum_desc_axes")
      check_onnx_test_case!("node", "test_reduce_log_sum_negative_axes")
    end

    test "ReduceLogSumExp" do
      check_onnx_test_case!("node", "test_reduce_log_sum_exp_default_axes_keepdims_example")
      check_onnx_test_case!("node", "test_reduce_log_sum_exp_default_axes_keepdims_random")
      check_onnx_test_case!("node", "test_reduce_log_sum_exp_do_not_keepdims_example")
      check_onnx_test_case!("node", "test_reduce_log_sum_exp_do_not_keepdims_random")
      check_onnx_test_case!("node", "test_reduce_log_sum_exp_keepdims_example")
      check_onnx_test_case!("node", "test_reduce_log_sum_exp_keepdims_random")
      check_onnx_test_case!("node", "test_reduce_log_sum_exp_negative_axes_keepdims_example")
      check_onnx_test_case!("node", "test_reduce_log_sum_exp_negative_axes_keepdims_random")
    end

    test "ReduceL1" do
      check_onnx_test_case!("node", "test_reduce_l1_default_axes_keepdims_example")
      check_onnx_test_case!("node", "test_reduce_l1_default_axes_keepdims_random")
      check_onnx_test_case!("node", "test_reduce_l1_do_not_keepdims_example")
      check_onnx_test_case!("node", "test_reduce_l1_do_not_keepdims_random")
      check_onnx_test_case!("node", "test_reduce_l1_keep_dims_example")
      check_onnx_test_case!("node", "test_reduce_l1_keep_dims_random")
      check_onnx_test_case!("node", "test_reduce_l1_negative_axes_keep_dims_example")
      check_onnx_test_case!("node", "test_reduce_l1_negative_axes_keep_dims_random")
    end

    # test "ReduceL2" do
    #   check_onnx_test_case!("node", "test_reduce_l2_default_axes_keepdims_example")
    #   check_onnx_test_case!("node", "test_reduce_l2_default_axes_keepdims_random")
    #   check_onnx_test_case!("node", "test_reduce_l2_do_not_keepdims_example")
    #   check_onnx_test_case!("node", "test_reduce_l2_do_not_keepdims_random")
    #   check_onnx_test_case!("node", "test_reduce_l2_keepdims_example")
    #   check_onnx_test_case!("node", "test_reduce_l2_keepdims_random")
    #   check_onnx_test_case!("node", "test_reduce_l2_negative_axes_keep_dims_example")
    #   check_onnx_test_case!("node", "test_reduce_l2_negative_axes_keep_dims_random")
    # end

    test "ReduceMax" do
      check_onnx_test_case!("node", "test_reduce_max_default_axes_keepdim_example")
      check_onnx_test_case!("node", "test_reduce_max_default_axes_keepdims_random")
      check_onnx_test_case!("node", "test_reduce_max_do_not_keepdims_example")
      check_onnx_test_case!("node", "test_reduce_max_do_not_keepdims_random")
      check_onnx_test_case!("node", "test_reduce_max_keepdims_example")
      check_onnx_test_case!("node", "test_reduce_max_keepdims_random")
      check_onnx_test_case!("node", "test_reduce_max_negative_axes_keepdims_example")
      check_onnx_test_case!("node", "test_reduce_max_negative_axes_keepdims_random")
    end

    test "ReduceMean" do
      check_onnx_test_case!("node", "test_reduce_mean_default_axes_keepdims_example")
      check_onnx_test_case!("node", "test_reduce_mean_default_axes_keepdims_random")
      check_onnx_test_case!("node", "test_reduce_mean_do_not_keepdims_example")
      check_onnx_test_case!("node", "test_reduce_mean_do_not_keepdims_random")
      check_onnx_test_case!("node", "test_reduce_mean_keepdims_example")
      check_onnx_test_case!("node", "test_reduce_mean_keepdims_random")
      check_onnx_test_case!("node", "test_reduce_mean_negative_axes_keepdims_example")
      check_onnx_test_case!("node", "test_reduce_mean_negative_axes_keepdims_random")
    end

    test "ReduceMin" do
      check_onnx_test_case!("node", "test_reduce_min_default_axes_keepdims_example")
      check_onnx_test_case!("node", "test_reduce_min_default_axes_keepdims_random")
      check_onnx_test_case!("node", "test_reduce_min_do_not_keepdims_example")
      check_onnx_test_case!("node", "test_reduce_min_do_not_keepdims_random")
      check_onnx_test_case!("node", "test_reduce_min_keepdims_example")
      check_onnx_test_case!("node", "test_reduce_min_keepdims_random")
      check_onnx_test_case!("node", "test_reduce_min_negative_axes_keepdims_example")
      check_onnx_test_case!("node", "test_reduce_min_negative_axes_keepdims_random")
    end

    test "ReduceProd" do
      check_onnx_test_case!("node", "test_reduce_prod_default_axes_keepdims_example")
      check_onnx_test_case!("node", "test_reduce_prod_default_axes_keepdims_random")
      check_onnx_test_case!("node", "test_reduce_prod_do_not_keepdims_example")
      check_onnx_test_case!("node", "test_reduce_prod_do_not_keepdims_random")
      check_onnx_test_case!("node", "test_reduce_prod_keepdims_example")
      check_onnx_test_case!("node", "test_reduce_prod_keepdims_random")
      check_onnx_test_case!("node", "test_reduce_prod_negative_axes_keepdims_example")
      check_onnx_test_case!("node", "test_reduce_prod_negative_axes_keepdims_random")
    end

    test "ReduceSumSquare" do
      check_onnx_test_case!("node", "test_reduce_sum_square_default_axes_keepdims_example")
      check_onnx_test_case!("node", "test_reduce_sum_square_default_axes_keepdims_random")
      check_onnx_test_case!("node", "test_reduce_sum_square_do_not_keepdims_example")
      check_onnx_test_case!("node", "test_reduce_sum_square_do_not_keepdims_random")
      check_onnx_test_case!("node", "test_reduce_sum_square_keepdims_example")
      check_onnx_test_case!("node", "test_reduce_sum_square_keepdims_random")
      check_onnx_test_case!("node", "test_reduce_sum_square_negative_axes_keepdims_example")
      check_onnx_test_case!("node", "test_reduce_sum_square_negative_axes_keepdims_random")
    end

    test "Relu" do
      check_onnx_test_case!("node", "test_relu")
    end

    # TODO: Dynamic Shapes :(
    # test "Reshape" do
    # check_onnx_test_case!("node", "test_reshape_allowzero_reordered")
    # check_onnx_test_case!("node", "test_reshape_extended_dims")
    # check_onnx_test_case!("node", "test_reshape_negative_dims")
    # check_onnx_test_case!("node", "test_reshape_negative_extended_dims")
    # check_onnx_test_case!("node", "test_reshape_one_dim")
    # check_onnx_test_case!("node", "test_reshape_reduced_dims")
    # check_onnx_test_case!("node", "test_reshape_reordered_all_dims")
    # check_onnx_test_case!("node", "test_reshape_reordered_last_dims")
    # check_onnx_test_case!("node", "test_reshape_zero_and_negative_dims")
    # check_onnx_test_case!("node", "test_reshape_zero_dim")
    # end

    @tag :resize
    test "Resize" do
      check_onnx_test_case!("node", "test_resize_downsample_scales_cubic")
      # check_onnx_test_case!("node", "test_resize_downsample_scales_cubic_A_n0p5_exclude_outside")
      # check_onnx_test_case!("node", "test_resize_downsample_scales_cubic_align_corners")
      # check_onnx_test_case!("node", "test_resize_downsample_scales_linear")
      # check_onnx_test_case!("node", "test_resize_downsample_scales_linear_align_corners")
      # check_onnx_test_case!("node", "test_resize_downsample_scales_nearest")
      # check_onnx_test_case!("node", "test_resize_downsample_sizes_cubic")
      # check_onnx_test_case!("node", "test_resize_downsample_sizes_linear_pytorch_half_pixel")
      # check_onnx_test_case!("node", "test_resize_downsample_sizes_nearest")
      # check_onnx_test_case!("node", "test_resize_downsample_sizes_nearest_tf_half_pixel_for_nn")
      # check_onnx_test_case!("node", "test_resize_tf_crop_and_resize")
      # check_onnx_test_case!("node", "test_resize_upsample_scales_cubic")
      # check_onnx_test_case!("node", "test_resize_upsample_scales_cubic_A_n0p5_exclude_outside")
      # check_onnx_test_case!("node", "test_resize_upsample_scales_cubic_align_corners")
      # check_onnx_test_case!("node", "test_resize_upsample_scales_cubic_asymmetric")
      # check_onnx_test_case!("node", "test_resize_upsample_scales_linear")
      # check_onnx_test_case!("node", "test_resize_upsample_scales_linear_align_corners")
      # check_onnx_test_case!("node", "test_resize_upsample_scales_nearest")
      # check_onnx_test_case!("node", "test_resize_upsample_scales_nearest_ceil_half_pixel")
      # check_onnx_test_case!("node", "test_resize_upsample_scales_nearest_floor_align_corners")
      # check_onnx_test_case!("node", "test_resize_upsample_scales_nearest_round_prefer_ceil_asymmetric")
    end

    # test "Round" do
    #   check_onnx_test_case!("node", "test_round")
    # end

    test "Selu" do
      check_onnx_test_case!("node", "test_selu")
      check_onnx_test_case!("node", "test_selu_default")
      check_onnx_test_case!("node", "test_selu_example")
    end

    test "Shape" do
      check_onnx_test_case!("node", "test_shape")
      check_onnx_test_case!("node", "test_shape_clip_end")
      check_onnx_test_case!("node", "test_shape_clip_start")
      check_onnx_test_case!("node", "test_shape_end_1")
      check_onnx_test_case!("node", "test_shape_end_negative_1")
      check_onnx_test_case!("node", "test_shape_example")
      check_onnx_test_case!("node", "test_shape_start_1")
      check_onnx_test_case!("node", "test_shape_start_1_end_2")
      check_onnx_test_case!("node", "test_shape_start_1_end_negative_1")
      check_onnx_test_case!("node", "test_shape_start_negative_1")
    end

    test "Sigmoid" do
      check_onnx_test_case!("node", "test_sigmoid")
      check_onnx_test_case!("node", "test_sigmoid_example")
    end

    test "Sign" do
      check_onnx_test_case!("node", "test_sign")
    end

    test "Sin" do
      check_onnx_test_case!("node", "test_sin")
      check_onnx_test_case!("node", "test_sin_example")
    end

    test "Sinh" do
      check_onnx_test_case!("node", "test_sinh")
      check_onnx_test_case!("node", "test_sinh_example")
    end

    test "Softmax" do
      check_onnx_test_case!("node", "test_softmax_axis_0")
      # TODO: ReduceMax
      # check_onnx_test_case!("node", "test_softmax_axis_0_expanded")
      check_onnx_test_case!("node", "test_softmax_axis_1")
      # TODO: ReduceMax
      # check_onnx_test_case!("node", "test_softmax_axis_1_expanded")
      check_onnx_test_case!("node", "test_softmax_axis_2")
      # TODO: ReduceMax
      # check_onnx_test_case!("node", "test_softmax_axis_2_expanded")
      check_onnx_test_case!("node", "test_softmax_default_axis")
      # TODO: ReduceMax
      # check_onnx_test_case!("node", "test_softmax_default_axis_expanded")
      check_onnx_test_case!("node", "test_softmax_large_number")
      # TODO: ReduceMax
      # check_onnx_test_case!("node", "test_softmax_large_number_expanded")
      check_onnx_test_case!("node", "test_softmax_negative_axis")
      # TODO: ReduceMax
      # check_onnx_test_case!("node", "test_softmax_negative_axis_expanded")
      check_onnx_test_case!("node", "test_softmax_example")
      # TODO: ReduceMax
      # check_onnx_test_case!("node", "test_softmax_example_expanded")
    end

    test "Softplus" do
      check_onnx_test_case!("node", "test_softplus")
      check_onnx_test_case!("node", "test_softplus_example")
    end

    test "Softsign" do
      check_onnx_test_case!("node", "test_softsign")
      check_onnx_test_case!("node", "test_softsign_example")
    end

    test "Sqrt" do
      check_onnx_test_case!("node", "test_sqrt")
      check_onnx_test_case!("node", "test_sqrt_example")
    end

    test "Sub" do
      check_onnx_test_case!("node", "test_sub")
      check_onnx_test_case!("node", "test_sub_bcast")
      check_onnx_test_case!("node", "test_sub_uint8")
    end

    test "Sum" do
      check_onnx_test_case!("node", "test_sum_example")
      check_onnx_test_case!("node", "test_sum_one_input")
      check_onnx_test_case!("node", "test_sum_two_inputs")
    end

    test "Tan" do
      check_onnx_test_case!("node", "test_tan")
      check_onnx_test_case!("node", "test_tan_example")
    end

    test "Tanh" do
      check_onnx_test_case!("node", "test_tanh")
      check_onnx_test_case!("node", "test_tanh_example")
    end

    test "Transpose" do
      check_onnx_test_case!("node", "test_transpose_default")
      check_onnx_test_case!("node", "test_transpose_all_permutations_0")
      check_onnx_test_case!("node", "test_transpose_all_permutations_1")
      check_onnx_test_case!("node", "test_transpose_all_permutations_2")
      check_onnx_test_case!("node", "test_transpose_all_permutations_3")
      check_onnx_test_case!("node", "test_transpose_all_permutations_4")
      check_onnx_test_case!("node", "test_transpose_all_permutations_5")
    end

    # TODO: Dynamic Shapes :(
    # test "Unsqueeze" do
    #   check_onnx_test_case!("node", "test_unsqueeze_axis_0")
    #   check_onnx_test_case!("node", "test_unsqueeze_axis_1")
    #   check_onnx_test_case!("node", "test_unsqueeze_axis_2")
    #   check_onnx_test_case!("node", "test_unsqueeze_axis_3")
    #   check_onnx_test_case!("node", "test_unsqueeze_negative_axes")
    #   check_onnx_test_case!("node", "test_unsqueeze_two_axes")
    #   check_onnx_test_case!("node", "test_unsqueeze_three_axes")
    #   check_onnx_test_case!("node", "test_unsqueeze_unsorted_axes")
    # end

    test "Xor" do
      check_onnx_test_case!("node", "test_xor2d")
      check_onnx_test_case!("node", "test_xor3d")
      check_onnx_test_case!("node", "test_xor4d")
      check_onnx_test_case!("node", "test_xor_bcast3v1d")
      check_onnx_test_case!("node", "test_xor_bcast3v2d")
      check_onnx_test_case!("node", "test_xor_bcast4v2d")
      check_onnx_test_case!("node", "test_xor_bcast4v3d")
      check_onnx_test_case!("node", "test_xor_bcast4v4d")
    end

    test "Where" do
      check_onnx_test_case!("node", "test_where_example")
      check_onnx_test_case!("node", "test_where_long_example")
    end
  end

  describe "pytorch converted tests" do
    test "AvgPool" do
      check_onnx_test_case!("pytorch-converted", "test_AvgPool1d")
      check_onnx_test_case!("pytorch-converted", "test_AvgPool1d_stride")
      check_onnx_test_case!("pytorch-converted", "test_AvgPool2d")
      check_onnx_test_case!("pytorch-converted", "test_AvgPool2d_stride")
      check_onnx_test_case!("pytorch-converted", "test_AvgPool3d")
      check_onnx_test_case!("pytorch-converted", "test_AvgPool3d_stride")
      check_onnx_test_case!("pytorch-converted", "test_AvgPool3d_stride1_pad0_gpu_input")
    end

    test "BatchNorm" do
      check_onnx_test_case!("pytorch-converted", "test_BatchNorm1d_3d_input_eval")
      check_onnx_test_case!("pytorch-converted", "test_BatchNorm2d_eval")
      check_onnx_test_case!("pytorch-converted", "test_BatchNorm2d_momentum_eval")
      check_onnx_test_case!("pytorch-converted", "test_BatchNorm3d_eval")
      check_onnx_test_case!("pytorch-converted", "test_BatchNorm3d_momentum_eval")
    end

    test "ConstantPad2d" do
      check_onnx_test_case!("pytorch-converted", "test_ConstantPad2d")
    end

    test "Conv" do
      check_onnx_test_case!("pytorch-converted", "test_Conv1d")
      check_onnx_test_case!("pytorch-converted", "test_Conv1d_dilated")
      check_onnx_test_case!("pytorch-converted", "test_Conv1d_groups")
      check_onnx_test_case!("pytorch-converted", "test_Conv1d_pad1")
      check_onnx_test_case!("pytorch-converted", "test_Conv1d_pad1size1")
      check_onnx_test_case!("pytorch-converted", "test_Conv1d_pad2")
      check_onnx_test_case!("pytorch-converted", "test_Conv1d_pad2size1")
      check_onnx_test_case!("pytorch-converted", "test_Conv1d_stride")
      check_onnx_test_case!("pytorch-converted", "test_Conv2d")
      check_onnx_test_case!("pytorch-converted", "test_Conv2d_depthwise")
      check_onnx_test_case!("pytorch-converted", "test_Conv2d_depthwise_padded")
      check_onnx_test_case!("pytorch-converted", "test_Conv2d_depthwise_strided")
      check_onnx_test_case!("pytorch-converted", "test_Conv2d_depthwise_with_multiplier")
      check_onnx_test_case!("pytorch-converted", "test_Conv2d_dilated")
      check_onnx_test_case!("pytorch-converted", "test_Conv2d_groups")
      check_onnx_test_case!("pytorch-converted", "test_Conv2d_groups_thnn")
      check_onnx_test_case!("pytorch-converted", "test_Conv2d_no_bias")
      check_onnx_test_case!("pytorch-converted", "test_Conv2d_padding")
      check_onnx_test_case!("pytorch-converted", "test_Conv2d_strided")
      check_onnx_test_case!("pytorch-converted", "test_Conv3d")
      check_onnx_test_case!("pytorch-converted", "test_Conv3d_dilated")
      check_onnx_test_case!("pytorch-converted", "test_Conv3d_dilated_strided")
      check_onnx_test_case!("pytorch-converted", "test_Conv3d_groups")
      check_onnx_test_case!("pytorch-converted", "test_Conv3d_no_bias")
      check_onnx_test_case!("pytorch-converted", "test_Conv3d_stride")
      check_onnx_test_case!("pytorch-converted", "test_Conv3d_stride_padding")
    end

    # test "ConvTranspose" do
    #   check_onnx_test_case!("pytorch-converted", "test_ConvTranspose2d")
    #   check_onnx_test_case!("pytorch-converted", "test_ConvTranspose2d_no_bias")
    # end

    test "ELU" do
      check_onnx_test_case!("pytorch-converted", "test_ELU")
    end

    test "Embedding" do
      check_onnx_test_case!("pytorch-converted", "test_Embedding")
      check_onnx_test_case!("pytorch-converted", "test_Embedding_sparse")
    end

    # test "GLU" do
    #   check_onnx_test_case!("pytorch-converted", "test_GLU")
    #   check_onnx_test_case!("pytorch-converted", "test_GLU_dim")
    # end

    test "LeakyReLU" do
      check_onnx_test_case!("pytorch-converted", "test_LeakyReLU")
      check_onnx_test_case!("pytorch-converted", "test_LeakyReLU_with_negval")
    end

    test "Linear" do
      check_onnx_test_case!("pytorch-converted", "test_Linear")
      check_onnx_test_case!("pytorch-converted", "test_Linear_no_bias")
    end

    test "LogSoftmax" do
      check_onnx_test_case!("pytorch-converted", "test_log_softmax_lastdim")
      check_onnx_test_case!("pytorch-converted", "test_log_softmax_dim3")
      check_onnx_test_case!("pytorch-converted", "test_LogSoftmax")
    end

    test "MaxPool" do
      check_onnx_test_case!("pytorch-converted", "test_MaxPool1d")
      check_onnx_test_case!("pytorch-converted", "test_MaxPool1d_stride")
      check_onnx_test_case!("pytorch-converted", "test_MaxPool1d_stride_padding_dilation")
      check_onnx_test_case!("pytorch-converted", "test_MaxPool2d")
      check_onnx_test_case!("pytorch-converted", "test_MaxPool2d_stride_padding_dilation")
      check_onnx_test_case!("pytorch-converted", "test_MaxPool3d")
      check_onnx_test_case!("pytorch-converted", "test_MaxPool3d_stride")
    end

    # test "PReLU" do
    #   check_onnx_test_case!("pytorch-converted", "test_PReLU_1d")
    #   check_onnx_test_case!("pytorch-converted", "test_PReLU_1d_multiparam")
    #   check_onnx_test_case!("pytorch-converted", "test_PReLU_2d")
    #   check_onnx_test_case!("pytorch-converted", "test_PReLU_2d_multiparam")
    #   check_onnx_test_case!("pytorch-converted", "test_PReLU_3d")
    #   check_onnx_test_case!("pytorch-converted", "test_PReLU_3d_multiparam")
    # end

    test "PixelShuffle" do
      check_onnx_test_case!("pytorch-converted", "test_PixelShuffle")
    end

    test "PoissonNLLLLoss" do
      check_onnx_test_case!("pytorch-converted", "test_PoissonNLLLLoss_no_reduce")
    end

    test "ReLU" do
      check_onnx_test_case!("pytorch-converted", "test_ReLU")
    end

    # test "ReflectionPad2d" do
    #   check_onnx_test_case!("pytorch-converted", "test_ReflectionPad2d")
    # end

    # test "ReplicationPad2d" do
    #   check_onnx_test_case!("pytorch-converted", "test_ReplicationPad2d")
    # end

    test "SELU" do
      check_onnx_test_case!("pytorch-converted", "test_SELU")
    end

    test "Sigmoid" do
      check_onnx_test_case!("pytorch-converted", "test_Sigmoid")
    end

    test "Softmax" do
      check_onnx_test_case!("pytorch-converted", "test_Softmax")
      check_onnx_test_case!("pytorch-converted", "test_softmax_functional_dim3")
      check_onnx_test_case!("pytorch-converted", "test_softmax_lastdim")
    end

    test "Softmin" do
      check_onnx_test_case!("pytorch-converted", "test_Softmin")
    end

    test "Softplus" do
      check_onnx_test_case!("pytorch-converted", "test_Softplus")
    end

    test "Softsign" do
      check_onnx_test_case!("pytorch-converted", "test_Softsign")
    end

    test "Tanh" do
      check_onnx_test_case!("pytorch-converted", "test_Tanh")
    end

    test "ZeroPad2d" do
      check_onnx_test_case!("pytorch-converted", "test_ZeroPad2d")
    end
  end

  describe "pytorch operator tests" do
    # test "add_broadcast" do
    #   check_onnx_test_case!("pytorch-operator", "test_operator_add_broadcast")
    # end

    # test "add_size1_broadcast" do
    #   check_onnx_test_case!("pytorch-operator", "test_operator_add_size1_broadcast")
    # end

    # test "add_size1_right_broadcast" do
    #   check_onnx_test_case!("pytorch-operator", "test_operator_add_size1_right_broadcast")
    # end

    # test "add_size1_singleton_broadcast" do
    #   check_onnx_test_case!("pytorch-operator", "test_operator_add_size1_singleton_broadcast")
    # end

    # test "addconstant" do
    #   check_onnx_test_case!("pytorch-operator", "test_operator_addconstant")
    # end

    test "addmm" do
      check_onnx_test_case!("pytorch-operator", "test_operator_addmm")
    end

    test "basic" do
      check_onnx_test_case!("pytorch-operator", "test_operator_basic")
    end

    # test "chunk" do
    #   check_onnx_test_case!("pytorch-operator", "test_operator_chunk")
    # end

    # test "clip" do
    #   check_onnx_test_case!("pytorch-operator", "test_operator_clip")
    # end

    test "concat2" do
      check_onnx_test_case!("pytorch-operator", "test_operator_concat2")
    end

    # test "conv" do
    #   Nx.Defn.default_options(compiler: EXLA)
    #   check_onnx_test_case!("pytorch-operator", "test_operator_conv")
    # end

    test "exp" do
      check_onnx_test_case!("pytorch-operator", "test_operator_exp")
    end

    test "flatten" do
      check_onnx_test_case!("pytorch-operator", "test_operator_flatten")
    end

    # test "index" do
    #   check_onnx_test_case!("pytorch-operator", "test_operator_index")
    # end

    # test "max" do
    #   check_onnx_test_case!("pytorch-operator", "test_operator_max")
    # end

    test "maxpool" do
      check_onnx_test_case!("pytorch-operator", "test_operator_maxpool")
    end

    # test "min" do
    #   check_onnx_test_case!("pytorch-operator", "test_operator_min")
    # end

    test "mm" do
      check_onnx_test_case!("pytorch-operator", "test_operator_mm")
    end

    test "non_float_params" do
      check_onnx_test_case!("pytorch-operator", "test_operator_non_float_params")
    end

    # test "pad" do
    #   check_onnx_test_case!("pytorch-operator", "test_operator_pad")
    # end

    test "params" do
      check_onnx_test_case!("pytorch-operator", "test_operator_params")
    end

    test "permute" do
      check_onnx_test_case!("pytorch-operator", "test_operator_permute2")
    end

    # test "pow" do
    #   check_onnx_test_case!("pytorch-operator", "test_operator_pow")
    # end

    test "reduce_mean" do
      check_onnx_test_case!("pytorch-operator", "test_operator_reduced_mean")
      check_onnx_test_case!("pytorch-operator", "test_operator_reduced_mean_keepdim")
    end

    # test "reduce_sum" do
    #   check_onnx_test_case!("pytorch-operator", "test_operator_reduced_sum")
    #   check_onnx_test_case!("pytorch-operator", "test_operator_reduced_sum_keepdim")
    # end

    # test "repeat" do
    #   check_onnx_test_case!("pytorch-operator", "test_operator_repeat")
    #   check_onnx_test_case!("pytorch-operator", "test_operator_repeat_dim_overflow")
    # end

    test "selu" do
      check_onnx_test_case!("pytorch-operator", "test_operator_selu")
    end

    # test "sqrt" do
    #   check_onnx_test_case!("pytorch-operator", "test_operator_sqrt")
    # end

    # test "symbolic" do
    #   check_onnx_test_case!("pytorch-operator", "test_operator_symbolic_override")
    #   check_onnx_test_case!("pytorch-operator", "test_operator_symbolic_override_nested")
    # end

    test "view" do
      check_onnx_test_case!("pytorch-operator", "test_operator_view")
    end
  end

  describe "simple tests" do
    # test "expand shape model" do
    #   check_onnx_test_case!("simple", "test_expand_shape_model1")
    #   check_onnx_test_case!("simple", "test_expand_shape_model2")
    #   check_onnx_test_case!("simple", "test_expand_shape_model3")
    #   check_onnx_test_case!("simple", "test_expand_shape_model4")
    # end

    # test "gradient model" do
    #   check_onnx_test_case!("simple", "test_gradient_of_add")
    #   check_onnx_test_case!("simple", "test_gradient_of_add_mul")
    # end

    # test "sequence model" do
    #   check_onnx_test_case!("simple", "test_sequence_model1")
    #   check_onnx_test_case!("simple", "test_sequence_model2")
    #   check_onnx_test_case!("simple", "test_sequence_model3")
    #   check_onnx_test_case!("simple", "test_sequence_model4")
    #   check_onnx_test_case!("simple", "test_sequence_model5")
    #   check_onnx_test_case!("simple", "test_sequence_model6")
    #   check_onnx_test_case!("simple", "test_sequence_model7")
    #   check_onnx_test_case!("simple", "test_sequence_model8")
    # end

    # test "shrink model" do
    #   check_onnx_test_case!("simple", "test_shrink")
    # end

    test "sign model" do
      check_onnx_test_case!("simple", "test_sign_model")
    end

    test "single relu model" do
      check_onnx_test_case!("simple", "test_single_relu_model")
    end

    # test "strnorm model" do
    #   check_onnx_test_case!("simple", "test_strnorm_model_monday_casesensintive_lower")
    #   check_onnx_test_case!("simple", "test_strnorm_model_monday_casesensintive_nochangecase")
    #   check_onnx_test_case!("simple", "test_strnorm_model_monday_casesensintive_upper")
    #   check_onnx_test_case!("simple", "test_strnorm_model_monday_empty_output")
    #   check_onnx_test_case!("simple", "test_strnorm_model_monday_insensitive_upper_twodim")
    #   check_onnx_test_case!("simple", "test_strnorm_model_nostopwords_nochangecase")
    # end
  end

  describe "real tests" do
    @describetag :real
    @describetag timeout: 120_000

    # test "bvlc alexnet" do
    #   Nx.Defn.default_options(compiler: EXLA)
    #   check_onnx_model!("bvlc_alexnet")
    # end

    test "densenet121" do
      check_onnx_model!("densenet121", compiler: EXLA)
    end

    # test "inception_v1" do
    #   Nx.Defn.default_options(compiler: EXLA)
    #   check_onnx_model!("inception_v1")
    # end

    # test "inception_v2" do
    #   Nx.Defn.default_options(compiler: EXLA)
    #   check_onnx_model!("inception_v2")
    # end

    test "resnet50" do
      check_onnx_model!("resnet50", compiler: EXLA)
    end

    test "shufflenet" do
      check_onnx_model!("shufflenet", compiler: EXLA)
    end

    # test "squeezenet" do
    #   Nx.Defn.default_options(compiler: EXLA)
    #   check_onnx_model!("squeezenet")
    # end

    # TODO: This is too slow for ci
    # test "vgg19" do
    #   check_onnx_model!("vgg19", compiler: EXLA)
    # end

    # test "zfnet512" do
    #   Nx.Defn.default_options(compiler: EXLA)
    #   check_onnx_model!("zfnet512")
    # end
  end

  describe "torchvision tests" do
    @describetag :torchvision
    @describetag timeout: 120_000

    test "alexnet" do
      check_onnx_test_case!("torchvision", "alexnet", compiler: EXLA)
    end

    test "convnext_tiny" do
      check_onnx_test_case!("torchvision", "convnext_tiny", compiler: EXLA)
    end

    test "convnext_small" do
      check_onnx_test_case!("torchvision", "convnext_small", compiler: EXLA)
    end

    test "convnext_base" do
      check_onnx_test_case!("torchvision", "convnext_base", compiler: EXLA)
    end

    test "convnext_large" do
      check_onnx_test_case!("torchvision", "convnext_large", compiler: EXLA)
    end

    test "efficientnet_b0" do
      check_onnx_test_case!("torchvision", "efficientnet_b0", compiler: EXLA)
    end

    test "efficientnet_b1" do
      check_onnx_test_case!("torchvision", "efficientnet_b1", compiler: EXLA)
    end

    test "efficientnet_b2" do
      check_onnx_test_case!("torchvision", "efficientnet_b2", compiler: EXLA)
    end

    test "efficientnet_b3" do
      check_onnx_test_case!("torchvision", "efficientnet_b3", compiler: EXLA)
    end

    test "efficientnet_b4" do
      check_onnx_test_case!("torchvision", "efficientnet_b4", compiler: EXLA)
    end

    test "efficientnet_b5" do
      check_onnx_test_case!("torchvision", "efficientnet_b5", compiler: EXLA)
    end

    test "efficientnet_b6" do
      check_onnx_test_case!("torchvision", "efficientnet_b6", compiler: EXLA)
    end

    test "efficientnet_b7" do
      check_onnx_test_case!("torchvision", "efficientnet_b7", compiler: EXLA)
    end

    test "efficientnet_v2_s" do
      check_onnx_test_case!("torchvision", "efficientnet_v2_s", compiler: EXLA)
    end

    test "efficientnet_v2_m" do
      check_onnx_test_case!("torchvision", "efficientnet_v2_s", compiler: EXLA)
    end

    test "efficientnet_v2_l" do
      check_onnx_test_case!("torchvision", "efficientnet_v2_s", compiler: EXLA)
    end

    test "vgg11" do
      check_onnx_test_case!("torchvision", "vgg11", compiler: EXLA)
    end

    test "vgg11_bn" do
      check_onnx_test_case!("torchvision", "vgg11_bn", compiler: EXLA)
    end

    test "vgg13" do
      check_onnx_test_case!("torchvision", "vgg13", compiler: EXLA)
    end

    test "vgg13_bn" do
      check_onnx_test_case!("torchvision", "vgg13_bn", compiler: EXLA)
    end

    test "vgg16" do
      check_onnx_test_case!("torchvision", "vgg16", compiler: EXLA)
    end

    test "vgg16_bn" do
      check_onnx_test_case!("torchvision", "vgg16_bn", compiler: EXLA)
    end

    test "vgg19" do
      check_onnx_test_case!("torchvision", "vgg19", compiler: EXLA)
    end

    test "vgg19_bn" do
      check_onnx_test_case!("torchvision", "vgg19_bn", compiler: EXLA)
    end

    test "resnet18" do
      check_onnx_test_case!("torchvision", "resnet18", compiler: EXLA)
    end

    test "resnet34" do
      check_onnx_test_case!("torchvision", "resnet34", compiler: EXLA)
    end

    test "resnet50" do
      check_onnx_test_case!("torchvision", "resnet50", compiler: EXLA)
    end

    test "resnet101" do
      check_onnx_test_case!("torchvision", "resnet101", compiler: EXLA)
    end

    test "resnet152" do
      check_onnx_test_case!("torchvision", "resnet152", compiler: EXLA)
    end

    test "squeezenet1_0" do
      check_onnx_test_case!("torchvision", "squeezenet1_0", compiler: EXLA)
    end

    test "squeezenet1_1" do
      check_onnx_test_case!("torchvision", "squeezenet1_1", compiler: EXLA)
    end

    test "densenet121" do
      check_onnx_test_case!("torchvision", "densenet121", compiler: EXLA)
    end

    test "densenet161" do
      check_onnx_test_case!("torchvision", "densenet121", compiler: EXLA)
    end

    test "densenet169" do
      check_onnx_test_case!("torchvision", "densenet121", compiler: EXLA)
    end

    test "densenet201" do
      check_onnx_test_case!("torchvision", "densenet121", compiler: EXLA)
    end

    test "inception_v3" do
      check_onnx_test_case!("torchvision", "inception_v3", compiler: EXLA)
    end

    test "googlenet" do
      check_onnx_test_case!("torchvision", "googlenet", compiler: EXLA)
    end

    test "shufflenet_v2_x0_5" do
      check_onnx_test_case!("torchvision", "shufflenet_v2_x0_5", compiler: EXLA)
    end

    test "shufflenet_v2_x1_0" do
      check_onnx_test_case!("torchvision", "shufflenet_v2_x1_0", compiler: EXLA)
    end

    test "shufflenet_v2_x1_5" do
      check_onnx_test_case!("torchvision", "shufflenet_v2_x1_5", compiler: EXLA)
    end

    test "shufflenet_v2_x2_0" do
      check_onnx_test_case!("torchvision", "shufflenet_v2_x2_0", compiler: EXLA)
    end

    test "mobilenet_v2" do
      check_onnx_test_case!("torchvision", "mobilenet_v2", compiler: EXLA)
    end

    test "mobilenet_v3_small" do
      check_onnx_test_case!("torchvision", "mobilenet_v3_small", compiler: EXLA)
    end

    test "mobilenet_v3_large" do
      check_onnx_test_case!("torchvision", "mobilenet_v3_large", compiler: EXLA)
    end

    test "resnext50_32x4d" do
      check_onnx_test_case!("torchvision", "resnext50_32x4d", compiler: EXLA)
    end

    test "resnext101_32x8d" do
      check_onnx_test_case!("torchvision", "resnext101_32x8d", compiler: EXLA)
    end

    test "resnext101_64x4d" do
      check_onnx_test_case!("torchvision", "resnext101_64x4d", compiler: EXLA)
    end

    test "wide_resnet50_2" do
      check_onnx_test_case!("torchvision", "wide_resnet50_2", compiler: EXLA)
    end

    test "wide_resnet101_2" do
      check_onnx_test_case!("torchvision", "wide_resnet101_2", compiler: EXLA)
    end

    test "mnasnet0_5" do
      check_onnx_test_case!("torchvision", "mnasnet0_5", compiler: EXLA)
    end

    test "mnasnet0_75" do
      check_onnx_test_case!("torchvision", "mnasnet0_75", compiler: EXLA)
    end

    test "mnasnet1_0" do
      check_onnx_test_case!("torchvision", "mnasnet1_0", compiler: EXLA)
    end

    test "mnasnet1_3" do
      check_onnx_test_case!("torchvision", "mnasnet1_3", compiler: EXLA)
    end

    # TODO: Slicing bug
    # test "swin_t" do
    #   check_onnx_test_case!("torchvision", "swin_t", compiler: EXLA)
    # end

    # test "swin_s" do
    #   check_onnx_test_case!("torchvision", "swin_s", compiler: EXLA)
    # end

    # test "swin_b" do
    #   check_onnx_test_case!("torchvision", "swin_b", compiler: EXLA)
    # end

    # TODO: Gather bug
    # test "vit_b_16" do
    #   check_onnx_test_case!("torchvision", "vit_b_16", compiler: EXLA)
    # end

    # test "vit_b_32" do
    #   check_onnx_test_case!("torchvision", "vit_b_32", compiler: EXLA)
    # end

    # test "vit_l_16" do
    #   check_onnx_test_case!("torchvision", "vit_l_16", compiler: EXLA)
    # end

    # test "vit_l_32" do
    #   check_onnx_test_case!("torchvision", "vit_l_32", compiler: EXLA)
    # end

    # test "vit_h_14" do
    #   check_onnx_test_case!("torchvision", "vit_h_14", compiler: EXLA)
    # end
  end

  describe "transformer tests" do
    @describetag :transformers
    @describetag timeout: :infinity, capture_log: true

    test "albert" do
      check_onnx_transformer!("albert-base-v2", batch: 2, sequence: 8)
    end

    test "bart" do
      check_onnx_transformer!("facebook/bart-base",
        batch: 2,
        encoder_sequence: 8,
        decoder_sequence: 8
      )
    end

    test "beit" do
      check_onnx_transformer!("microsoft/beit-base-patch16-224", batch: 2, sequence: 8)
    end

    test "bert" do
      check_onnx_transformer!("bert-base-cased", batch: 2, sequence: 8)
    end

    test "bigbird" do
      check_onnx_transformer!("google/bigbird-roberta-base", batch: 2, sequence: 8)
    end

    test "blenderbot" do
      check_onnx_transformer!("facebook/blenderbot-400M-distill",
        batch: 2,
        encoder_sequence: 8,
        decoder_sequence: 8
      )
    end

    test "blenderbot_small" do
      check_onnx_transformer!("facebook/blenderbot_small-90M",
        batch: 2,
        encoder_sequence: 8,
        decoder_sequence: 8
      )
    end

    test "camembert" do
      check_onnx_transformer!("camembert-base", batch: 2, sequence: 8)
    end

    # TODO: This does not seem right...
    # test "convbert" do
    #   check_onnx_transformer!("YituTech/conv-bert-small", batch: 2, sequence: 8)
    # end

    test "data2vectext" do
      check_onnx_transformer!("facebook/data2vec-text-base", batch: 2, sequence: 8)
    end

    test "deit" do
      check_onnx_transformer!("facebook/deit-base-patch16-224", batch: 2, sequence: 8)
    end

    test "distilbert" do
      check_onnx_transformer!("distilbert-base-cased", batch: 2, sequence: 8)
    end

    test "electra" do
      check_onnx_transformer!("google/electra-small-discriminator", batch: 2, sequence: 8)
      check_onnx_transformer!("google/electra-small-generator", batch: 2, sequence: 8)
    end

    # TODO: For some reason this does not appear in CI
    # test "flaubert" do
    #   check_onnx_transformer!("flaubert/flaubert_small_cased", batch: 2, sequence: 8)
    # end

    # TODO: This model is 22.5GB so not the easiest to test
    # in a single GH action
    # test "gpt j" do
    #   check_onnx_transformer!("EleutherAI/gpt-j-6B")
    # end

    test "gpt neo" do
      check_onnx_transformer!("EleutherAI/gpt-neo-125M", batch: 2, sequence: 8)
    end

    test "gpt2" do
      check_onnx_transformer!("gpt2", batch: 2, sequence: 8)
    end

    test "layoutlm" do
      check_onnx_transformer!("microsoft/layoutlm-base-cased", batch: 2, sequence: 8)
    end

    test "m2m100" do
      check_onnx_transformer!("facebook/m2m100_418M",
        batch: 2,
        encoder_sequence: 8,
        decoder_sequence: 8
      )
    end

    # test "mbart" do
    #   check_onnx_transformer!("facebook/mbart-large-50")
    # end

    # TODO: Listed as supported, but does not appear to be?
    # test "plbart" do
    #   check_onnx_transformer!("uclanlp/plbart-base")
    # end

    test "roberta" do
      check_onnx_transformer!("roberta-base", batch: 2, sequence: 8)
    end

    test "t5" do
      check_onnx_transformer!("t5-base", batch: 2, encoder_sequence: 8, decoder_sequence: 8)
    end

    test "vit" do
      check_onnx_transformer!("google/vit-base-patch16-224", batch: 2, sequence: 8)
    end

    test "xlm-roberta" do
      check_onnx_transformer!("xlm-roberta-base", batch: 2, sequence: 8)
    end
  end
end
