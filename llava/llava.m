Network LLaVA_ViT_L14_336_CONV {
	Layer PATCH_EMBED {
		Type: CONV
		Stride { X: 14, Y: 14 }
		Dimensions { K: 1024, C: 3, R: 14, S: 14, Y: 336, X: 336 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK01_QKV {
		Type: CONV
		Dimensions { K: 3072, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK01_ATTN_QKT {
		Type: CONV
		Dimensions { K: 577, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK01_ATTN_AV {
		Type: CONV
		Dimensions { K: 1024, C: 577, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK01_ATTN_PROJ {
		Type: CONV
		Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK01_MLP_FC1 {
		Type: CONV
		Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK01_MLP_FC2 {
		Type: CONV
		Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK02_QKV {
		Type: CONV
		Dimensions { K: 3072, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK02_ATTN_QKT {
		Type: CONV
		Dimensions { K: 577, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK02_ATTN_AV {
		Type: CONV
		Dimensions { K: 1024, C: 577, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK02_ATTN_PROJ {
		Type: CONV
		Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK02_MLP_FC1 {
		Type: CONV
		Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK02_MLP_FC2 {
		Type: CONV
		Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK03_QKV {
		Type: CONV
		Dimensions { K: 3072, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK03_ATTN_QKT {
		Type: CONV
		Dimensions { K: 577, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK03_ATTN_AV {
		Type: CONV
		Dimensions { K: 1024, C: 577, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK03_ATTN_PROJ {
		Type: CONV
		Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK03_MLP_FC1 {
		Type: CONV
		Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK03_MLP_FC2 {
		Type: CONV
		Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK04_QKV {
		Type: CONV
		Dimensions { K: 3072, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK04_ATTN_QKT {
		Type: CONV
		Dimensions { K: 577, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK04_ATTN_AV {
		Type: CONV
		Dimensions { K: 1024, C: 577, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK04_ATTN_PROJ {
		Type: CONV
		Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK04_MLP_FC1 {
		Type: CONV
		Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK04_MLP_FC2 {
		Type: CONV
		Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK05_QKV {
		Type: CONV
		Dimensions { K: 3072, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK05_ATTN_QKT {
		Type: CONV
		Dimensions { K: 577, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK05_ATTN_AV {
		Type: CONV
		Dimensions { K: 1024, C: 577, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK05_ATTN_PROJ {
		Type: CONV
		Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK05_MLP_FC1 {
		Type: CONV
		Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK05_MLP_FC2 {
		Type: CONV
		Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK06_QKV {
		Type: CONV
		Dimensions { K: 3072, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK06_ATTN_QKT {
		Type: CONV
		Dimensions { K: 577, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK06_ATTN_AV {
		Type: CONV
		Dimensions { K: 1024, C: 577, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK06_ATTN_PROJ {
		Type: CONV
		Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK06_MLP_FC1 {
		Type: CONV
		Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK06_MLP_FC2 {
		Type: CONV
		Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK07_QKV {
		Type: CONV
		Dimensions { K: 3072, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK07_ATTN_QKT {
		Type: CONV
		Dimensions { K: 577, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK07_ATTN_AV {
		Type: CONV
		Dimensions { K: 1024, C: 577, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK07_ATTN_PROJ {
		Type: CONV
		Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK07_MLP_FC1 {
		Type: CONV
		Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK07_MLP_FC2 {
		Type: CONV
		Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK08_QKV {
		Type: CONV
		Dimensions { K: 3072, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK08_ATTN_QKT {
		Type: CONV
		Dimensions { K: 577, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK08_ATTN_AV {
		Type: CONV
		Dimensions { K: 1024, C: 577, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK08_ATTN_PROJ {
		Type: CONV
		Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK08_MLP_FC1 {
		Type: CONV
		Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK08_MLP_FC2 {
		Type: CONV
		Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK09_QKV {
		Type: CONV
		Dimensions { K: 3072, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK09_ATTN_QKT {
		Type: CONV
		Dimensions { K: 577, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK09_ATTN_AV {
		Type: CONV
		Dimensions { K: 1024, C: 577, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK09_ATTN_PROJ {
		Type: CONV
		Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK09_MLP_FC1 {
		Type: CONV
		Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK09_MLP_FC2 {
		Type: CONV
		Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK10_QKV {
		Type: CONV
		Dimensions { K: 3072, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK10_ATTN_QKT {
		Type: CONV
		Dimensions { K: 577, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK10_ATTN_AV {
		Type: CONV
		Dimensions { K: 1024, C: 577, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK10_ATTN_PROJ {
		Type: CONV
		Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK10_MLP_FC1 {
		Type: CONV
		Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK10_MLP_FC2 {
		Type: CONV
		Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK11_QKV {
		Type: CONV
		Dimensions { K: 3072, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK11_ATTN_QKT {
		Type: CONV
		Dimensions { K: 577, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK11_ATTN_AV {
		Type: CONV
		Dimensions { K: 1024, C: 577, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK11_ATTN_PROJ {
		Type: CONV
		Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK11_MLP_FC1 {
		Type: CONV
		Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK11_MLP_FC2 {
		Type: CONV
		Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK12_QKV {
		Type: CONV
		Dimensions { K: 3072, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK12_ATTN_QKT {
		Type: CONV
		Dimensions { K: 577, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK12_ATTN_AV {
		Type: CONV
		Dimensions { K: 1024, C: 577, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK12_ATTN_PROJ {
		Type: CONV
		Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK12_MLP_FC1 {
		Type: CONV
		Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK12_MLP_FC2 {
		Type: CONV
		Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK13_QKV {
		Type: CONV
		Dimensions { K: 3072, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK13_ATTN_QKT {
		Type: CONV
		Dimensions { K: 577, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK13_ATTN_AV {
		Type: CONV
		Dimensions { K: 1024, C: 577, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK13_ATTN_PROJ {
		Type: CONV
		Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK13_MLP_FC1 {
		Type: CONV
		Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK13_MLP_FC2 {
		Type: CONV
		Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK14_QKV {
		Type: CONV
		Dimensions { K: 3072, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK14_ATTN_QKT {
		Type: CONV
		Dimensions { K: 577, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK14_ATTN_AV {
		Type: CONV
		Dimensions { K: 1024, C: 577, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK14_ATTN_PROJ {
		Type: CONV
		Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK14_MLP_FC1 {
		Type: CONV
		Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK14_MLP_FC2 {
		Type: CONV
		Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK15_QKV {
		Type: CONV
		Dimensions { K: 3072, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK15_ATTN_QKT {
		Type: CONV
		Dimensions { K: 577, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK15_ATTN_AV {
		Type: CONV
		Dimensions { K: 1024, C: 577, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK15_ATTN_PROJ {
		Type: CONV
		Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK15_MLP_FC1 {
		Type: CONV
		Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK15_MLP_FC2 {
		Type: CONV
		Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK16_QKV {
		Type: CONV
		Dimensions { K: 3072, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK16_ATTN_QKT {
		Type: CONV
		Dimensions { K: 577, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK16_ATTN_AV {
		Type: CONV
		Dimensions { K: 1024, C: 577, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK16_ATTN_PROJ {
		Type: CONV
		Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK16_MLP_FC1 {
		Type: CONV
		Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK16_MLP_FC2 {
		Type: CONV
		Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK17_QKV {
		Type: CONV
		Dimensions { K: 3072, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK17_ATTN_QKT {
		Type: CONV
		Dimensions { K: 577, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK17_ATTN_AV {
		Type: CONV
		Dimensions { K: 1024, C: 577, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK17_ATTN_PROJ {
		Type: CONV
		Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK17_MLP_FC1 {
		Type: CONV
		Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK17_MLP_FC2 {
		Type: CONV
		Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK18_QKV {
		Type: CONV
		Dimensions { K: 3072, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK18_ATTN_QKT {
		Type: CONV
		Dimensions { K: 577, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK18_ATTN_AV {
		Type: CONV
		Dimensions { K: 1024, C: 577, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK18_ATTN_PROJ {
		Type: CONV
		Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK18_MLP_FC1 {
		Type: CONV
		Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK18_MLP_FC2 {
		Type: CONV
		Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK19_QKV {
		Type: CONV
		Dimensions { K: 3072, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK19_ATTN_QKT {
		Type: CONV
		Dimensions { K: 577, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK19_ATTN_AV {
		Type: CONV
		Dimensions { K: 1024, C: 577, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK19_ATTN_PROJ {
		Type: CONV
		Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK19_MLP_FC1 {
		Type: CONV
		Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK19_MLP_FC2 {
		Type: CONV
		Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK20_QKV {
		Type: CONV
		Dimensions { K: 3072, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK20_ATTN_QKT {
		Type: CONV
		Dimensions { K: 577, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK20_ATTN_AV {
		Type: CONV
		Dimensions { K: 1024, C: 577, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK20_ATTN_PROJ {
		Type: CONV
		Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK20_MLP_FC1 {
		Type: CONV
		Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK20_MLP_FC2 {
		Type: CONV
		Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK21_QKV {
		Type: CONV
		Dimensions { K: 3072, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK21_ATTN_QKT {
		Type: CONV
		Dimensions { K: 577, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK21_ATTN_AV {
		Type: CONV
		Dimensions { K: 1024, C: 577, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK21_ATTN_PROJ {
		Type: CONV
		Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK21_MLP_FC1 {
		Type: CONV
		Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK21_MLP_FC2 {
		Type: CONV
		Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK22_QKV {
		Type: CONV
		Dimensions { K: 3072, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK22_ATTN_QKT {
		Type: CONV
		Dimensions { K: 577, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK22_ATTN_AV {
		Type: CONV
		Dimensions { K: 1024, C: 577, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK22_ATTN_PROJ {
		Type: CONV
		Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK22_MLP_FC1 {
		Type: CONV
		Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK22_MLP_FC2 {
		Type: CONV
		Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK23_QKV {
		Type: CONV
		Dimensions { K: 3072, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK23_ATTN_QKT {
		Type: CONV
		Dimensions { K: 577, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK23_ATTN_AV {
		Type: CONV
		Dimensions { K: 1024, C: 577, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK23_ATTN_PROJ {
		Type: CONV
		Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK23_MLP_FC1 {
		Type: CONV
		Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK23_MLP_FC2 {
		Type: CONV
		Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK24_QKV {
		Type: CONV
		Dimensions { K: 3072, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK24_ATTN_QKT {
		Type: CONV
		Dimensions { K: 577, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK24_ATTN_AV {
		Type: CONV
		Dimensions { K: 1024, C: 577, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK24_ATTN_PROJ {
		Type: CONV
		Dimensions { K: 1024, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK24_MLP_FC1 {
		Type: CONV
		Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer BLK24_MLP_FC2 {
		Type: CONV
		Dimensions { K: 1024, C: 4096, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

	Layer VISION_TO_LLM_PROJ {
		Type: CONV
		Dimensions { K: 4096, C: 1024, R: 1, S: 1, Y: 577, X: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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
	
	Layer LLM_GEMM0 {
	        Type: CONV
	        Dimensions { K: 2048, C: 4096, Y: 128, X: 1, R: 1, S: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

    	Layer LLM_GEMM1 {
        	Type: CONV
        	Dimensions { K: 2048, C: 4096, Y: 128, X: 1, R: 1, S: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

    	Layer LLM_GEMM2 {
        	Type: CONV
        	Dimensions { K: 3072, C: 4096, Y: 320, X: 1, R: 1, S: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

    	Layer LLM_GEMM3 {
        	Type: CONV
        	Dimensions { K: 2048, C: 4096, Y: 128, X: 1, R: 1, S: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

    	Layer LLM_GEMM4 {
        	Type: CONV
        	Dimensions { K: 2048, C: 4096, Y: 128, X: 1, R: 1, S: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

    	Layer LLM_GEMM5 {
        	Type: CONV
        	Dimensions { K: 3072, C: 4096, Y: 320, X: 1, R: 1, S: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

    	Layer LLM_GEMM6 {
        	Type: CONV
        	Dimensions { K: 3072, C: 4096, Y: 320, X: 1, R: 1, S: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

    	Layer LLM_GEMM7 {
        	Type: CONV
        	Dimensions { K: 3072, C: 4096, Y: 320, X: 1, R: 1, S: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

    	Layer LLM_GEMM8 {
        	Type: CONV
        	Dimensions { K: 3072, C: 4096, Y: 320, X: 1, R: 1, S: 1 }
                Dataflow {
                        // This is a NVDLA-like dataflow
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

}