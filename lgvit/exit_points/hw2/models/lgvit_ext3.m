Network DeiTHighway_Exit2 {
    Layer Embed_Conv { // Patch Embeddings: Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
        Type: CONV
        Stride { X: 16, Y: 16 }
        Dimensions { K: 768, C: 3, R: 16, S: 16, Y: 224, X: 224 }
    }

    // === Encoder Layers (0-8) ===
    // --- Layers 0-7 ---
    Layer Enc0_Attn_Q { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 768, C: 768, R: 1, S: 1, Y: 197, X: 1 } 
    }
    Layer Enc0_Attn_K { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 768, C: 768, R: 1, S: 1, Y: 197, X: 1 } 
    }
    Layer Enc0_Attn_V { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 768, C: 768, R: 1, S: 1, Y: 197, X: 1 } 
    }
    Layer Enc0_Attn_Output { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 768, C: 768, R: 1, S: 1, Y: 197, X: 1 } 
    }
    Layer Enc0_FFN1 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 3072, C: 768, R: 1, S: 1, Y: 197, X: 1 } 
    }
    Layer Enc0_FFN2 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 768, C: 3072, R: 1, S: 1, Y: 197, X: 1 } 
    }
    Layer Enc1_Attn_Q { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 768, C: 768, R: 1, S: 1, Y: 197, X: 1 } 
    }
    Layer Enc1_Attn_K { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 768, C: 768, R: 1, S: 1, Y: 197, X: 1 } 
    }
    Layer Enc1_Attn_V { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 768, C: 768, R: 1, S: 1, Y: 197, X: 1 } 
    }
    Layer Enc1_Attn_Output { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 768, C: 768, R: 1, S: 1, Y: 197, X: 1 } 
    }
    Layer Enc1_FFN1 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 3072, C: 768, R: 1, S: 1, Y: 197, X: 1 } 
    }
    Layer Enc1_FFN2 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 768, C: 3072, R: 1, S: 1, Y: 197, X: 1 } 
    }
    Layer Enc2_Attn_Q { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 768, C: 768, R: 1, S: 1, Y: 197, X: 1 } 
    }
    Layer Enc2_Attn_K { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 768, C: 768, R: 1, S: 1, Y: 197, X: 1 } 
    }
    Layer Enc2_Attn_V { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 768, C: 768, R: 1, S: 1, Y: 197, X: 1 } 
    }
    Layer Enc2_Attn_Output { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 768, C: 768, R: 1, S: 1, Y: 197, X: 1 } 
    }
    Layer Enc2_FFN1 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 3072, C: 768, R: 1, S: 1, Y: 197, X: 1 } 
    }
    Layer Enc2_FFN2 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 768, C: 3072, R: 1, S: 1, Y: 197, X: 1 } 
    }
    Layer Enc3_Attn_Q { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 768, C: 768, R: 1, S: 1, Y: 197, X: 1 } 
    }
    Layer Enc3_Attn_K { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 768, C: 768, R: 1, S: 1, Y: 197, X: 1 } 
    }
    Layer Enc3_Attn_V { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 768, C: 768, R: 1, S: 1, Y: 197, X: 1 } 
    }
    Layer Enc3_Attn_Output { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 768, C: 768, R: 1, S: 1, Y: 197, X: 1 } 
    }
    Layer Enc3_FFN1 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 3072, C: 768, R: 1, S: 1, Y: 197, X: 1 } 
    }
    Layer Enc3_FFN2 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 768, C: 3072, R: 1, S: 1, Y: 197, X: 1 } 
    }
    Layer Enc4_Attn_Q { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 768, C: 768, R: 1, S: 1, Y: 197, X: 1 } 
    }
    Layer Enc4_Attn_K { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 768, C: 768, R: 1, S: 1, Y: 197, X: 1 } 
    }
    Layer Enc4_Attn_V { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 768, C: 768, R: 1, S: 1, Y: 197, X: 1 } 
    }
    Layer Enc4_Attn_Output { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 768, C: 768, R: 1, S: 1, Y: 197, X: 1 } 
    }
    Layer Enc4_FFN1 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 3072, C: 768, R: 1, S: 1, Y: 197, X: 1 } 
    }
    Layer Enc4_FFN2 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 768, C: 3072, R: 1, S: 1, Y: 197, X: 1 } 
    }
    Layer Enc5_Attn_Q { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 768, C: 768, R: 1, S: 1, Y: 197, X: 1 } 
    }
    Layer Enc5_Attn_K { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 768, C: 768, R: 1, S: 1, Y: 197, X: 1 } 
    }
    Layer Enc5_Attn_V { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 768, C: 768, R: 1, S: 1, Y: 197, X: 1 } 
    }
    Layer Enc5_Attn_Output { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 768, C: 768, R: 1, S: 1, Y: 197, X: 1 } 
    }
    Layer Enc5_FFN1 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 3072, C: 768, R: 1, S: 1, Y: 197, X: 1 } 
    }
    Layer Enc5_FFN2 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 768, C: 3072, R: 1, S: 1, Y: 197, X: 1 } 
    }
    Layer Enc6_Attn_Q { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 768, C: 768, R: 1, S: 1, Y: 197, X: 1 } 
    }
    Layer Enc6_Attn_K { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 768, C: 768, R: 1, S: 1, Y: 197, X: 1 } 
    }
    Layer Enc6_Attn_V { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 768, C: 768, R: 1, S: 1, Y: 197, X: 1 } 
    }
    Layer Enc6_Attn_Output { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 768, C: 768, R: 1, S: 1, Y: 197, X: 1 } 
    }
    Layer Enc6_FFN1 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 3072, C: 768, R: 1, S: 1, Y: 197, X: 1 } 
    }
    Layer Enc6_FFN2 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 768, C: 3072, R: 1, S: 1, Y: 197, X: 1 } 
    }
    Layer Enc7_Attn_Q { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 768, C: 768, R: 1, S: 1, Y: 197, X: 1 } 
    }
    Layer Enc7_Attn_K { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 768, C: 768, R: 1, S: 1, Y: 197, X: 1 } 
    }
    Layer Enc7_Attn_V { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 768, C: 768, R: 1, S: 1, Y: 197, X: 1 } 
    }
    Layer Enc7_Attn_Output { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 768, C: 768, R: 1, S: 1, Y: 197, X: 1 } 
    }
    Layer Enc7_FFN1 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 3072, C: 768, R: 1, S: 1, Y: 197, X: 1 } 
    }
    Layer Enc7_FFN2 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 768, C: 3072, R: 1, S: 1, Y: 197, X: 1 } 
    }
    // --- Layer 8 ---
    Layer Enc8_Attn_Q { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 768, C: 768, R: 1, S: 1, Y: 197, X: 1 } 
    }
    Layer Enc8_Attn_K { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 768, C: 768, R: 1, S: 1, Y: 197, X: 1 } 
    }
    Layer Enc8_Attn_V { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 768, C: 768, R: 1, S: 1, Y: 197, X: 1 } 
    }
    Layer Enc8_Attn_Output { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 768, C: 768, R: 1, S: 1, Y: 197, X: 1 } 
    }
    Layer Enc8_FFN1 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 3072, C: 768, R: 1, S: 1, Y: 197, X: 1 } 
    }
    Layer Enc8_FFN2 { 
        Type: CONV 
        Stride { X: 1, Y: 1 } 
        Dimensions { K: 768, C: 3072, R: 1, S: 1, Y: 197, X: 1 } 
    }
    
    // === Highway Exit 2 (after Encoder Layer 8) ===
    Layer Highway2_Pooler { // Represents: Pooler Linear(768, 768)
        Type: CONV
        Stride { X: 1, Y: 1 }
        Dimensions { K: 768, C: 768, R: 1, S: 1, Y: 1, X: 1 }
    }
}
