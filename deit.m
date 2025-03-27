Network DeiTForImageClassificationWithTeacher {

Layer PatchEmbedding {
    Type: CONV
    Stride { X: 16, Y: 16 }
    Dimensions { K: 768, C: 3, R: 16, S: 16, Y: 224, X: 224 }
    Dataflow {
        SpatialMap(1,1) K;
        TemporalMap(64,64) C;
        TemporalMap(Sz(R),Sz(R)) R;
        TemporalMap(Sz(S),Sz(S)) S;
        TemporalMap(Sz(R),1) Y;
        TemporalMap(Sz(S),1) X;
        Cluster(64, P);
        SpatialMap(1,1) C;
        TemporalMap(Sz(R),1) Y;
        TemporalMap(Sz(S),1) X;
        TemporalMap(Sz(R),Sz(R)) R;
        TemporalMap(Sz(S),Sz(S)) S;
    }
}

Layer TransformerAttention {
    Type: GEMM
    Dimensions { M: 768, K: 768, N: 768 }
    Dataflow {
        SpatialMap(1,1) M;
        TemporalMap(64,64) K;
        TemporalMap(64,64) N;
        Cluster(64, P);
    }
}

Layer MLP_1 {
    Type: GEMM
    Dimensions { M: 768, K: 768, N: 3072 }
    Dataflow {
        SpatialMap(1,1) M;
        TemporalMap(64,64) K;
        TemporalMap(64,64) N;
        Cluster(64, P);
    }
}

Layer MLP_2 {
    Type: GEMM
    Dimensions { M: 3072, K: 3072, N: 768 }
    Dataflow {
        SpatialMap(1,1) M;
        TemporalMap(64,64) K;
        TemporalMap(64,64) N;
        Cluster(64, P);
    }
}

Layer ClassificationHead {
    Type: GEMM
    Dimensions { M: 768, K: 768, N: 1000 }
    Dataflow {
        SpatialMap(1,1) M;
        TemporalMap(64,64) K;
        TemporalMap(64,64) N;
        Cluster(64, P);
    }
}

}
