// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Arguments are:
// XNN_DWCONV_UNIPASS(arch, name, first_pass_tile, middle_pass_tile, last_pass_tile, channel_tile, channel_subtile, channel_round, datatype, weights_type, buffer_type,params_type, init_fn)

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon, xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neon, 5, 5, 5, 4, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon, xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neon_acc2, 5, 5, 5, 4, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon_fma, xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neonfma, 5, 5, 5, 4, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon_fma, xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neonfma_acc2, 5, 5, 5, 4, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon, xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neon, 5, 5, 5, 8, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon, xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neon_acc2, 5, 5, 5, 8, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon_fma, xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neonfma, 5, 5, 5, 8, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon_fma, xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neonfma_acc2, 5, 5, 5, 8, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon, xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neon, 6, 6, 7, 4, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon, xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neon_acc2, 6, 6, 7, 4, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon_fma, xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neonfma, 6, 6, 7, 4, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon_fma, xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neonfma_acc2, 6, 6, 7, 4, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon, xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neon, 6, 6, 7, 8, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon, xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neon_acc2, 6, 6, 7, 8, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon_fma, xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neonfma, 6, 6, 7, 8, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon_fma, xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neonfma_acc2, 6, 6, 7, 8, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon, xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neon, 8, 8, 9, 4, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon, xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neon_acc2, 8, 8, 9, 4, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon_fma, xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neonfma, 8, 8, 9, 4, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon_fma, xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neonfma_acc2, 8, 8, 9, 4, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon, xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neon, 8, 8, 9, 8, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon, xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neon_acc2, 8, 8, 9, 8, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon_fma, xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neonfma, 8, 8, 9, 8, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_arm_neon_fma, xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neonfma_acc2, 8, 8, 9, 8, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__sse, 5, 5, 5, 4, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__sse_acc2, 5, 5, 5, 4, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__sse, 5, 5, 5, 8, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__sse_acc2, 5, 5, 5, 8, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_minmax_ukernel_5f5m5l16c4s4r__sse, 5, 5, 5, 16, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_minmax_ukernel_5f5m5l16c4s4r__sse_acc2, 5, 5, 5, 16, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__sse, 6, 6, 7, 4, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__sse_acc2, 6, 6, 7, 4, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__sse, 6, 6, 7, 8, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__sse_acc2, 6, 6, 7, 8, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_minmax_ukernel_6f6m7l16c4s4r__sse, 6, 6, 7, 16, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_minmax_ukernel_6f6m7l16c4s4r__sse_acc2, 6, 6, 7, 16, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__sse, 8, 8, 9, 4, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__sse_acc2, 8, 8, 9, 4, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__sse, 8, 8, 9, 8, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__sse_acc2, 8, 8, 9, 8, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_minmax_ukernel_8f8m9l16c4s4r__sse, 8, 8, 9, 16, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_minmax_ukernel_8f8m9l16c4s4r__sse_acc2, 8, 8, 9, 16, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_x86_avx, xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__avx, 5, 5, 5, 8, 8, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_x86_avx, xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__avx_acc2, 5, 5, 5, 8, 8, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_x86_avx, xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__avx, 5, 5, 5, 16, 8, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_x86_avx, xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__avx_acc2, 5, 5, 5, 16, 8, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_x86_avx, xnn_f32_dwconv_minmax_ukernel_6f6m7l8c8s4r__avx, 6, 6, 7, 8, 8, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_x86_avx, xnn_f32_dwconv_minmax_ukernel_6f6m7l8c8s4r__avx_acc2, 6, 6, 7, 8, 8, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_x86_avx, xnn_f32_dwconv_minmax_ukernel_6f6m7l16c8s4r__avx, 6, 6, 7, 16, 8, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_x86_avx, xnn_f32_dwconv_minmax_ukernel_6f6m7l16c8s4r__avx_acc2, 6, 6, 7, 16, 8, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_x86_avx, xnn_f32_dwconv_minmax_ukernel_8f8m9l8c8s4r__avx, 8, 8, 9, 8, 8, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_x86_avx, xnn_f32_dwconv_minmax_ukernel_8f8m9l8c8s4r__avx_acc2, 8, 8, 9, 8, 8, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_x86_avx, xnn_f32_dwconv_minmax_ukernel_8f8m9l16c8s4r__avx, 8, 8, 9, 16, 8, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_x86_avx, xnn_f32_dwconv_minmax_ukernel_8f8m9l16c8s4r__avx_acc2, 8, 8, 9, 16, 8, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_x86_fma3, xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__fma3, 5, 5, 5, 8, 8, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_x86_fma3, xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__fma3_acc2, 5, 5, 5, 8, 8, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_x86_fma3, xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__fma3, 5, 5, 5, 16, 8, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_x86_fma3, xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__fma3_acc2, 5, 5, 5, 16, 8, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_x86_fma3, xnn_f32_dwconv_minmax_ukernel_5f5m5l32c8s4r__fma3, 5, 5, 5, 32, 8, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_x86_fma3, xnn_f32_dwconv_minmax_ukernel_5f5m5l32c8s4r__fma3_acc2, 5, 5, 5, 32, 8, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_x86_fma3, xnn_f32_dwconv_minmax_ukernel_7f6m6l8c8s4r__fma3, 7, 6, 6, 8, 8, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_x86_fma3, xnn_f32_dwconv_minmax_ukernel_7f6m6l8c8s4r__fma3_acc2, 7, 6, 6, 8, 8, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_x86_fma3, xnn_f32_dwconv_minmax_ukernel_7f6m6l16c8s4r__fma3, 7, 6, 6, 16, 8, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_x86_fma3, xnn_f32_dwconv_minmax_ukernel_7f6m6l16c8s4r__fma3_acc2, 7, 6, 6, 16, 8, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_x86_fma3, xnn_f32_dwconv_minmax_ukernel_7f6m6l32c8s4r__fma3, 7, 6, 6, 32, 8, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_x86_fma3, xnn_f32_dwconv_minmax_ukernel_7f6m6l32c8s4r__fma3_acc2, 7, 6, 6, 32, 8, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ENABLE_AVX512F && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_DWCONV_MULTIPASS(xnn_arch_x86_avx512f, xnn_f32_dwconv_minmax_ukernel_5f5m5l16c16s1r__avx512f, 5, 5, 5, 16, 16, 1, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_x86_avx512f, xnn_f32_dwconv_minmax_ukernel_5f5m5l16c16s1r__avx512f_acc2, 5, 5, 5, 16, 16, 1, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_x86_avx512f, xnn_f32_dwconv_minmax_ukernel_5f5m5l32c16s1r__avx512f, 5, 5, 5, 32, 16, 1, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(xnn_arch_x86_avx512f, xnn_f32_dwconv_minmax_ukernel_5f5m5l32c16s1r__avx512f_acc2, 5, 5, 5, 32, 16, 1, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_minmax_ukernel_3f3m3l4c4s4r__wasmsimd_arm, 3, 3, 3, 4, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_minmax_ukernel_3f3m3l4c4s4r__wasmsimd_arm_acc2, 3, 3, 3, 4, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_minmax_ukernel_3f3m3l8c4s4r__wasmsimd_arm, 3, 3, 3, 8, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_minmax_ukernel_3f3m3l8c4s4r__wasmsimd_arm_acc2, 3, 3, 3, 8, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_minmax_ukernel_3f3m3l4c4s4r__wasmsimd_x86, 3, 3, 3, 4, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_minmax_ukernel_3f3m3l4c4s4r__wasmsimd_x86_acc2, 3, 3, 3, 4, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_minmax_ukernel_3f3m3l8c4s4r__wasmsimd_x86, 3, 3, 3, 8, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_minmax_ukernel_3f3m3l8c4s4r__wasmsimd_x86_acc2, 3, 3, 3, 8, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_arm, 5, 5, 5, 4, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_arm_acc2, 5, 5, 5, 4, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_x86, 5, 5, 5, 4, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_x86_acc2, 5, 5, 5, 4, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASMRELAXEDSIMD
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd, 5, 5, 5, 4, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_acc2, 5, 5, 5, 4, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma, 5, 5, 5, 4, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma_acc2, 5, 5, 5, 4, 4, 4, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
#endif  // XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_minmax_ukernel_3f3m3l1c1s1r__wasm, 3, 3, 3, 1, 1, 1, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_minmax_ukernel_3f3m3l1c1s1r__wasm_acc2, 3, 3, 3, 1, 1, 1, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__wasm, 5, 5, 5, 1, 1, 1, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__wasm_acc2, 5, 5, 5, 1, 1, 1, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__wasm, 6, 6, 7, 1, 1, 1, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__wasm_acc2, 6, 6, 7, 1, 1, 1, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__wasm, 8, 8, 9, 1, 1, 1, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__wasm_acc2, 8, 8, 9, 1, 1, 1, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_minmax_ukernel_2f2m2l1c1s1r__scalar, 2, 2, 2, 1, 1, 1, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_minmax_ukernel_2f2m2l1c1s1r__scalar_acc2, 2, 2, 2, 1, 1, 1, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_minmax_ukernel_2f2m2l4c1s1r__scalar, 2, 2, 2, 4, 1, 1, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_minmax_ukernel_2f2m2l4c1s1r__scalar_acc2, 2, 2, 2, 4, 1, 1, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__scalar, 5, 5, 5, 1, 1, 1, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__scalar_acc2, 5, 5, 5, 1, 1, 1, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__scalar, 6, 6, 7, 1, 1, 1, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__scalar_acc2, 6, 6, 7, 1, 1, 1, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__scalar, 8, 8, 9, 1, 1, 1, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_DWCONV_MULTIPASS(0, xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__scalar_acc2, 8, 8, 9, 1, 1, 1, float, float, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)

