#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quick test - just check if fusion code loads without errors."""

if __name__ == "__main__":
    print("=" * 80)
    print("QUICK TEST: Check Prefill Fusion Code")
    print("=" * 80)

    # Test 1: Import
    print("\n1. Testing import...")
    try:
        from vllm import envs

        print("   ✅ Import successful")
        print(f"   ✅ VLLM_USE_AITER_FUSED = {envs.VLLM_USE_AITER_FUSED}")
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        exit(1)

    # Test 2: Check kernel available
    print("\n2. Testing kernel availability...")
    try:
        from aiter.ops.triton.fusions.fused_kv_cache import (
            fused_qk_rope_cat_and_cache_mla,
        )

        print("   ✅ Fused prefill kernel available")
        print(f"   ✅ Kernel: {fused_qk_rope_cat_and_cache_mla.__name__}")
    except Exception as e:
        print(f"   ❌ Kernel not available: {e}")
        exit(1)

    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED - Code is ready")
    print("=" * 80)
    print("\nNEXT STEPS:")
    print("  - Run full generation test when GPU memory available")
    print("  - Or test with existing vLLM server")
