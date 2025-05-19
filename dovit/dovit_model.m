Network DoVit_Maestro {
    // (backbone): VisionTransformer -> (patch_embed): PatchEmbed -> (projection): Conv2d(3, 1024, kernel_size=(16, 16), stride=(16, 16))
    Layer PatchEmbed_Projection {
        Type: CONV
        Stride { X: 16, Y: 16 }
        Dimensions { K: 1024, C: 3, R: 16, S: 16, Y: 224, X: 224 } // K=out_channels, C=in_channels, R=kernel_h, S=kernel_w, Y=input_h, X=input_w
    }
    // --- Backbone Layer 0 ---
    // (attn): MultiheadAttention -> Q, K, V, Out projections (Linear(1024, 1024))
    Layer BB_L0_Attn_Q_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L0_Attn_K_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L0_Attn_V_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L0_Attn_Out_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    } // (attn.attn.out_proj)
    // (ffn): FFN -> Linear(1024, 4096) then Linear(4096, 1024)
    Layer BB_L0_FFN1 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 196,  X: 1 } 
    }
    Layer BB_L0_FFN2 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 196, X: 1 } 
    }

    // --- Backbone Layer 1 ---
    Layer BB_L1_Attn_Q_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L1_Attn_K_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L1_Attn_V_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L1_Attn_Out_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L1_FFN1 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 196,  X: 1 } 
    }
    Layer BB_L1_FFN2 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 196, X: 1 } 
    }

    // --- Backbone Layer 2 ---
    Layer BB_L2_Attn_Q_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L2_Attn_K_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L2_Attn_V_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L2_Attn_Out_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L2_FFN1 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 196,  X: 1 } 
    }
    Layer BB_L2_FFN2 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 196, X: 1 } 
    }

    // --- Backbone Layer 3 ---
    Layer BB_L3_Attn_Q_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L3_Attn_K_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L3_Attn_V_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L3_Attn_Out_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L3_FFN1 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 196,  X: 1 } 
    }
    Layer BB_L3_FFN2 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 196, X: 1 } 
    }

    // --- Backbone Layer 4 ---
    Layer BB_L4_Attn_Q_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L4_Attn_K_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L4_Attn_V_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L4_Attn_Out_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L4_FFN1 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 196,  X: 1 } 
    }
    Layer BB_L4_FFN2 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 196, X: 1 } 
    }

    // --- Backbone Layer 5 --- (Output feeds into AuxHead1)
    Layer BB_L5_Attn_Q_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L5_Attn_K_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L5_Attn_V_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L5_Attn_Out_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L5_FFN1 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 196,  X: 1 } 
    }
    Layer BB_L5_FFN2 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 196, X: 1 } 
    }

    //--------------------------------------------------------------------------
    // EXIT POINT 1: AuxiliaryHead1 (after Backbone Layer 5)
    // Input: (SeqLen, 1024) reshaped to (1024, H_patch, W_patch) = (1024, 14, 14)
    // (conv_seg): Conv2d(1024, 30, kernel_size=(1, 1), stride=(1, 1))
    //--------------------------------------------------------------------------
    Layer AuxHead1_ConvSeg {
        Type: CONV
        Stride { X: 1, Y: 1 }
        Dimensions { K: 30, C: 1024, R: 1, S: 1, Y: 14, X: 14 } // K=num_classes, C=embed_dim, Y=H_patch, X=W_patch
    }

    // --- Backbone Layer 6 ---
    Layer BB_L6_Attn_Q_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L6_Attn_K_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L6_Attn_V_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L6_Attn_Out_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L6_FFN1 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 196,  X: 1 } 
    }
    Layer BB_L6_FFN2 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 196, X: 1 } 
    }

    // --- Backbone Layer 7 ---
    Layer BB_L7_Attn_Q_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L7_Attn_K_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L7_Attn_V_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L7_Attn_Out_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L7_FFN1 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 196,  X: 1 } 
    }
    Layer BB_L7_FFN2 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 196, X: 1 } 
    }

    // --- Backbone Layer 8 ---
    Layer BB_L8_Attn_Q_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L8_Attn_K_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L8_Attn_V_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L8_Attn_Out_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L8_FFN1 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 196,  X: 1 } 
    }
    Layer BB_L8_FFN2 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 196, X: 1 } 
    }

    // --- Backbone Layer 9 ---
    Layer BB_L9_Attn_Q_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L9_Attn_K_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L9_Attn_V_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L9_Attn_Out_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L9_FFN1 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 196,  X: 1 } 
    }
    Layer BB_L9_FFN2 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 196, X: 1 } 
    }

    // --- Backbone Layer 10 ---
    Layer BB_L10_Attn_Q_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L10_Attn_K_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L10_Attn_V_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L10_Attn_Out_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L10_FFN1 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 196,  X: 1 } 
    }
    Layer BB_L10_FFN2 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 196, X: 1 } 
    }

    // --- Backbone Layer 11 --- (Output feeds into AuxHead2)
    Layer BB_L11_Attn_Q_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L11_Attn_K_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L11_Attn_V_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L11_Attn_Out_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L11_FFN1 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 196,  X: 1 } 
    }
    Layer BB_L11_FFN2 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 196, X: 1 } 
    }

    //--------------------------------------------------------------------------
    // EXIT POINT 2: AuxiliaryHead2 (after Backbone Layer 11)
    // (conv_seg): Conv2d(1024, 30, kernel_size=(1, 1), stride=(1, 1))
    //--------------------------------------------------------------------------
    Layer AuxHead2_ConvSeg {
        Type: CONV
        Stride { X: 1, Y: 1 }
        Dimensions { K: 30, C: 1024, R: 1, S: 1, Y: 14, X: 14 }
    }

    // --- Backbone Layer 12 ---
    Layer BB_L12_Attn_Q_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L12_Attn_K_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L12_Attn_V_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L12_Attn_Out_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L12_FFN1 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 196,  X: 1 } 
    }
    Layer BB_L12_FFN2 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 196, X: 1 } 
    }

    // --- Backbone Layer 13 ---
    Layer BB_L13_Attn_Q_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L13_Attn_K_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L13_Attn_V_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L13_Attn_Out_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L13_FFN1 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 196,  X: 1 } 
    }
    Layer BB_L13_FFN2 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 196, X: 1 } 
    }

    // --- Backbone Layer 14 ---
    Layer BB_L14_Attn_Q_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L14_Attn_K_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L14_Attn_V_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L14_Attn_Out_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L14_FFN1 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 196,  X: 1 } 
    }
    Layer BB_L14_FFN2 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 196, X: 1 } 
    }

    // --- Backbone Layer 15 ---
    Layer BB_L15_Attn_Q_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L15_Attn_K_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L15_Attn_V_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L15_Attn_Out_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L15_FFN1 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 196,  X: 1 } 
    }
    Layer BB_L15_FFN2 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 196, X: 1 } 
    }

    // --- Backbone Layer 16 ---
    Layer BB_L16_Attn_Q_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L16_Attn_K_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L16_Attn_V_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L16_Attn_Out_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L16_FFN1 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 196,  X: 1 } 
    }
    Layer BB_L16_FFN2 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 196, X: 1 } 
    }

    // --- Backbone Layer 17 --- (Output feeds into AuxHead3)
    Layer BB_L17_Attn_Q_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L17_Attn_K_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L17_Attn_V_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L17_Attn_Out_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L17_FFN1 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 196,  X: 1 } 
    }
    Layer BB_L17_FFN2 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 196, X: 1 } 
    }

    //--------------------------------------------------------------------------
    // EXIT POINT 3: AuxiliaryHead3 (after Backbone Layer 17)
    // (conv_seg): Conv2d(1024, 30, kernel_size=(1, 1), stride=(1, 1))
    //--------------------------------------------------------------------------
    Layer AuxHead3_ConvSeg {
        Type: CONV
        Stride { X: 1, Y: 1 }
        Dimensions { K: 30, C: 1024, R: 1, S: 1, Y: 14, X: 14 }
    }

    // --- Backbone Layer 18 ---
    Layer BB_L18_Attn_Q_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L18_Attn_K_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L18_Attn_V_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L18_Attn_Out_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L18_FFN1 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 196,  X: 1 } 
    }
    Layer BB_L18_FFN2 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 196, X: 1 } 
    }

    // --- Backbone Layer 19 ---
    Layer BB_L19_Attn_Q_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L19_Attn_K_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L19_Attn_V_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L19_Attn_Out_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L19_FFN1 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 196,  X: 1 } 
    }
    Layer BB_L19_FFN2 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 196, X: 1 } 
    }

    // --- Backbone Layer 20 ---
    Layer BB_L20_Attn_Q_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L20_Attn_K_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L20_Attn_V_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L20_Attn_Out_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L20_FFN1 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 196,  X: 1 } 
    }
    Layer BB_L20_FFN2 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 196, X: 1 } 
    }

    // --- Backbone Layer 21 ---
    Layer BB_L21_Attn_Q_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L21_Attn_K_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L21_Attn_V_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L21_Attn_Out_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L21_FFN1 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 196,  X: 1 } 
    }
    Layer BB_L21_FFN2 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 196, X: 1 } 
    }

    // --- Backbone Layer 22 ---
    Layer BB_L22_Attn_Q_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L22_Attn_K_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L22_Attn_V_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L22_Attn_Out_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L22_FFN1 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 196,  X: 1 } 
    }
    Layer BB_L22_FFN2 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 196, X: 1 } 
    }

    // --- Backbone Layer 23 --- (Final Backbone Layer, output feeds to DecodeHead)
    Layer BB_L23_Attn_Q_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L23_Attn_K_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L23_Attn_V_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L23_Attn_Out_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer BB_L23_FFN1 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 196,  X: 1 } 
    }
    Layer BB_L23_FFN2 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 196, X: 1 } 
    }

    //--------------------------------------------------------------------------
    // FINAL EXIT POINT: DecodeHead (SegmenterMaskTransformerHead)
    // Input: Output of Backbone Layer 23 (Sequence Length 196, Embed Dim 1024)
    //--------------------------------------------------------------------------

    // Decoder Transformer Layers (2 layers)
    // --- Decoder Layer 0 ---
    // (attn): MultiheadAttention -> Q, K, V, Out projections (Linear(1024, 1024))
    Layer Dec_L0_Attn_Q_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer Dec_L0_Attn_K_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer Dec_L0_Attn_V_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer Dec_L0_Attn_Out_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    } // (attn.out_proj)
    // (ffn): FFN -> Linear(1024, 4096) then Linear(4096, 1024)
    Layer Dec_L0_FFN1 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 196,  X: 1 } 
    }
    Layer Dec_L0_FFN2 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 196, X: 1 } 
    }

    // --- Decoder Layer 1 ---
    Layer Dec_L1_Attn_Q_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer Dec_L1_Attn_K_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer Dec_L1_Attn_V_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer Dec_L1_Attn_Out_Proj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    Layer Dec_L1_FFN1 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 196,  X: 1 } 
    }
    Layer Dec_L1_FFN2 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 196, X: 1 } 
    }

    // Decoder Head Projections
    // (dec_proj): Linear(in_features=1024, out_features=1024, bias=True)
    Layer DecHead_DecProj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    // (patch_proj): Linear(in_features=1024, out_features=1024, bias=False)
    Layer DecHead_PatchProj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }
    // (classes_proj): Linear(in_features=1024, out_features=1024, bias=False)
    Layer DecHead_ClassesProj { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 196, X: 1 } 
    }

    // Final Classifier for Decode Head (to get 30 classes)
    // Assumes output of previous layers (1024 features) is reshaped to (1024, H_patch, W_patch)
    // This layer is functionally equivalent to Conv2d(1024, 30, kernel_size=(1,1))
    Layer DecHead_FinalClassifier_Conv {
        Type: CONV
        Stride { X: 1, Y: 1 }
        Dimensions { K: 30, C: 1024, R: 1, S: 1, Y: 14, X: 14 } // K=num_classes, C=embed_dim, Y=H_patch, X=W_patch
    }
}
