Network transformers.models.deit_teacher {
    Layer Conv2d-1 {
        Type: CONV
        Stride { X: 16, Y: 16 }
        Dimensions { K: 768, C: 3, R: 16, S: 16, Y: 224, X: 224 }
    }

    // Transformer Encoder Layers: 12 blocks × 6 CONVs each = 72 layers
    // Each CONV replaces a GEMM following the rule: Y=M, K=N, C=K, X=R=S=1

    // Block 1
    Layer Conv2d-2 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 768, X: 1, R: 1, S: 1 } 
    } // Q
    Layer Conv2d-3 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 768, X: 1, R: 1, S: 1 } 
    } // K
    Layer Conv2d-4 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 768, X: 1, R: 1, S: 1 } 
    } // V
    Layer Conv2d-5 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 768, X: 1, R: 1, S: 1 } 
    } // attn output
    Layer Conv2d-6 { 
        Type: CONV 
        Dimensions { Y: 196, K: 3072, C: 768, X: 1, R: 1, S: 1 } 
    } // MLP 1
    Layer Conv2d-7 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 3072, X: 1, R: 1, S: 1 } 
    } // MLP 2

    // Block 2
    Layer Conv2d-8  { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 768, X: 1, R: 1, S: 1 } 
    }
    Layer Conv2d-9  { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 768, X: 1, R: 1, S: 1 } 
    }
    Layer Conv2d-10 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 768, X: 1, R: 1, S: 1 } 
    }
    Layer Conv2d-11 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 768, X: 1, R: 1, S: 1 } 
    }
    Layer Conv2d-12 { 
        Type: CONV 
        Dimensions { Y: 196, K: 3072, C: 768, X: 1, R: 1, S: 1 } 
    }
    Layer Conv2d-13 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 3072, X: 1, R: 1, S: 1 } 
    }

    // Repeat until Block 12

    // Block 3
    Layer Conv2d-14 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 768, X: 1, R: 1, S: 1 } 
    }
    Layer Conv2d-15 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 768, X: 1, R: 1, S: 1 } 
    }
    Layer Conv2d-16 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 768, X: 1, R: 1, S: 1 } 
    }
    Layer Conv2d-17 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 768, X: 1, R: 1, S: 1 } 
    }
    Layer Conv2d-18 { 
        Type: CONV 
        Dimensions { Y: 196, K: 3072, C: 768, X: 1, R: 1, S: 1 } 
    }
    Layer Conv2d-19 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 3072, X: 1, R: 1, S: 1 } 
    }

    // Block 4
    Layer Conv2d-20 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 768, X: 1, R: 1, S: 1 } 
    }
    Layer Conv2d-21 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 768, X: 1, R: 1, S: 1 } 
    }
    Layer Conv2d-22 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 768, X: 1, R: 1, S: 1 } 
    }
    Layer Conv2d-23 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 768, X: 1, R: 1, S: 1 } 
    }
    Layer Conv2d-24 { 
        Type: CONV 
        Dimensions { Y: 196, K: 3072, C: 768, X: 1, R: 1, S: 1 } 
    }
    Layer Conv2d-25 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 3072, X: 1, R: 1, S: 1 } 
    }

    // Block 5
    Layer Conv2d-26 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 768, X: 1, R: 1, S: 1 } 
    }
    Layer Conv2d-27 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 768, X: 1, R: 1, S: 1 } 
    }
    Layer Conv2d-28 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 768, X: 1, R: 1, S: 1 } 
    }
    Layer Conv2d-29 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 768, X: 1, R: 1, S: 1 } 
    }
    Layer Conv2d-30 { 
        Type: CONV 
        Dimensions { Y: 196, K: 3072, C: 768, X: 1, R: 1, S: 1 } 
    }
    Layer Conv2d-31 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 3072, X: 1, R: 1, S: 1 } 
    }

    // Block 6
    Layer Conv2d-32 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 768, X: 1, R: 1, S: 1 } 
    }
    Layer Conv2d-33 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 768, X: 1, R: 1, S: 1 } 
    }
    Layer Conv2d-34 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 768, X: 1, R: 1, S: 1 } 
    }
    Layer Conv2d-35 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 768, X: 1, R: 1, S: 1 } 
    }
    Layer Conv2d-36 { 
        Type: CONV 
        Dimensions { Y: 196, K: 3072, C: 768, X: 1, R: 1, S: 1 } 
    }
    Layer Conv2d-37 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 3072, X: 1, R: 1, S: 1 } 
    }

    // Block 7
    Layer Conv2d-38 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 768, X: 1, R: 1, S: 1 } 
    } // Q
    Layer Conv2d-39 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 768, X: 1, R: 1, S: 1 } 
    } // K
    Layer Conv2d-40 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 768, X: 1, R: 1, S: 1 } 
    } // V
    Layer Conv2d-41 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 768, X: 1, R: 1, S: 1 } 
    } // attn output
    Layer Conv2d-42 { 
        Type: CONV 
        Dimensions { Y: 196, K: 3072, C: 768, X: 1, R: 1, S: 1 } 
    } // MLP 1
    Layer Conv2d-43 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 3072, X: 1, R: 1, S: 1 } 
    } // MLP 2

    // Block 8
    Layer Conv2d-44  { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 768, X: 1, R: 1, S: 1 } 
    }
    Layer Conv2d-45  { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 768, X: 1, R: 1, S: 1 } 
    }
    Layer Conv2d-46 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 768, X: 1, R: 1, S: 1 } 
    }
    Layer Conv2d-47 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 768, X: 1, R: 1, S: 1 } 
    }
    Layer Conv2d-48 { 
        Type: CONV 
        Dimensions { Y: 196, K: 3072, C: 768, X: 1, R: 1, S: 1 } 
    }
    Layer Conv2d-49 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 3072, X: 1, R: 1, S: 1 } 
    }

    // Block 9
    Layer Conv2d-50 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 768, X: 1, R: 1, S: 1 } 
    }
    Layer Conv2d-51 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 768, X: 1, R: 1, S: 1 } 
    }
    Layer Conv2d-52 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 768, X: 1, R: 1, S: 1 } 
    }
    Layer Conv2d-53 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 768, X: 1, R: 1, S: 1 } 
    }
    Layer Conv2d-54 { 
        Type: CONV 
        Dimensions { Y: 196, K: 3072, C: 768, X: 1, R: 1, S: 1 } 
    }
    Layer Conv2d-55 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 3072, X: 1, R: 1, S: 1 } 
    }

    // Block 10
    Layer Conv2d-56 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 768, X: 1, R: 1, S: 1 } 
    }
    Layer Conv2d-57 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 768, X: 1, R: 1, S: 1 } 
    }
    Layer Conv2d-58 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 768, X: 1, R: 1, S: 1 } 
    }
    Layer Conv2d-59 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 768, X: 1, R: 1, S: 1 } 
    }
    Layer Conv2d-60 { 
        Type: CONV 
        Dimensions { Y: 196, K: 3072, C: 768, X: 1, R: 1, S: 1 } 
    }
    Layer Conv2d-61 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 3072, X: 1, R: 1, S: 1 } 
    }

    // Block 11
    Layer Conv2d-62 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 768, X: 1, R: 1, S: 1 } 
    }
    Layer Conv2d-63 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 768, X: 1, R: 1, S: 1 } 
    }
    Layer Conv2d-64 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 768, X: 1, R: 1, S: 1 } 
    }
    Layer Conv2d-65 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 768, X: 1, R: 1, S: 1 } 
    }
    Layer Conv2d-66 { 
        Type: CONV 
        Dimensions { Y: 196, K: 3072, C: 768, X: 1, R: 1, S: 1 } 
    }
    Layer Conv2d-67 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 3072, X: 1, R: 1, S: 1 } 
    }

    // Block 12
    Layer Conv2d-68 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 768, X: 1, R: 1, S: 1 } 
    }
    Layer Conv2d-69 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 768, X: 1, R: 1, S: 1 } 
    }
    Layer Conv2d-70 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 768, X: 1, R: 1, S: 1 } 
    }
    Layer Conv2d-71 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 768, X: 1, R: 1, S: 1 } 
    }
    Layer Conv2d-72 { 
        Type: CONV 
        Dimensions { Y: 196, K: 3072, C: 768, X: 1, R: 1, S: 1 } 
    }
    Layer Conv2d-73 { 
        Type: CONV 
        Dimensions { Y: 196, K: 768, C: 3072, X: 1, R: 1, S: 1 } 
    }

    // Classifier Heads (2 heads)
    Layer Conv2d-76 {
        Type: CONV
        Stride { X: 1, Y: 1 }
        Dimensions { K: 1000, C: 768, R: 1, S: 1, Y: 1, X: 1 }
    }

    Layer Conv2d-77 {
        Type: CONV
        Stride { X: 1, Y: 1 }
        Dimensions { K: 1000, C: 768, R: 1, S: 1, Y: 1, X: 1 }
    }
}
