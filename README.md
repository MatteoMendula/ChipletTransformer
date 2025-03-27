### DeiT Structure 
DeiTForImageClassificationWithTeacher, mainly consists of the following types of layers:

---

1. Convolutional Layer (Conv2D)
    
    Found in:
    ```
    DeiTPatchEmbeddings.projection (Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))).
    ```
    
    ✅ Supported in MAESTRO as CONV2D.

---

2. Fully Connected (FC) Layers

    Found in the classification heads (cls_classifier, distillation_classifier):
    ```
    Linear(in_features=768, out_features=1000, bias=True)
    ```
3. Also found in self-attention and MLP layers:
    ```
    Linear(in_features=768, out_features=768, bias=True)
    Linear(in_features=768, out_features=3072, bias=True)
    Linear(in_features=3072, out_features=768, bias=True)
    ```
    ✅ Supported indirectly via CONV2D following the specified rules (Y = R = 1, X = S, c = input features, k = output features).

3. Matrix Multiplication (GEMM) Operations

    Used in Transformer self-attention layers:
    ```
    Linear(in_features=768, out_features=768, bias=True)
    ```
    This represents QKV projections and dense layers in attention and MLPs.
    
    ✅ Supported in MAESTRO via GEMM.

---

### DeiT Main Components 
Since DeiT is a Vision Transformer (ViT)-based model, it primarily consists of:

1. Patch Embedding (Conv2D projection layer)
2. Self-Attention (GEMM)
3. MLP layers (GEMM)

Breakdown of Layers:
1. **Patch Embedding (Conv2D)**
    
    Projects input images (3x224x224) into 768-dimensional tokens using a 16x16 Conv2D.
    This is directly supported as a CONV2D layer in MAESTRO.

2. **Self-Attention (GEMM)**

    Implements matrix multiplication (GEMM) for self-attention.
    Uses the standard (M, K) x (K, N) → (M, N) format.
    
3. **MLP Layers (GEMM)**

    The first GEMM expands hidden states from 768 → 3072.
    The second GEMM projects them back from 3072 → 768.

4. **Classification Head (GEMM)**

    Final Linear (FC) layer mapping 768 → 1000 (ImageNet classes).


