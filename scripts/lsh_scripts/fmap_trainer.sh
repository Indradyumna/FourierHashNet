#SHORT VERSION
CUDA_VISIBLE_DEVICES=0 python -m lsh.train_fmaps dataset.name="msweb294"   dataset.rel_mode="fhash"  hashing.FUNC="sighinge" hashing.name="Fhash_Trained"  &
CUDA_VISIBLE_DEVICES=1 python -m lsh.train_fmaps dataset.name="msweb294_1" dataset.rel_mode="fhash"  hashing.FUNC="sighinge" hashing.name="Fhash_Trained"  &
CUDA_VISIBLE_DEVICES=2 python -m lsh.train_fmaps dataset.name="msnbc294_3" dataset.rel_mode="fhash"  hashing.FUNC="sighinge" hashing.name="Fhash_Trained"  &
CUDA_VISIBLE_DEVICES=3 python -m lsh.train_fmaps dataset.name="msnbc294_4" dataset.rel_mode="fhash"  hashing.FUNC="sighinge" hashing.name="Fhash_Trained"  &


#FULL VERSION -- run this if not hardcoded common values in config folder 
# CUDA_VISIBLE_DEVICES=3 python -m lsh.train_fmaps dataset.name="msweb294" dataset.rel_mode="fhash" hashing.m_use=10  dataset.embed_dim=294 hashing.T=3  fmap_training.tr_fmap_dim=10 fmap_training.hidden_layers=[]  hashing.FUNC="sighinge" hashing.name="Fhash_Trained" training.patience=50 fmap_training.tr_fmap_loss="BCE3"  fmap_training.margin=0.05
# CUDA_VISIBLE_DEVICES=3 python -m lsh.train_fmaps dataset.name="msweb294_1" dataset.rel_mode="fhash" hashing.m_use=10  dataset.embed_dim=294 hashing.T=3  fmap_training.tr_fmap_dim=10 fmap_training.hidden_layers=[]  hashing.FUNC="sighinge" hashing.name="Fhash_Trained" training.patience=50 fmap_training.tr_fmap_loss="BCE3"  fmap_training.margin=0.05
# CUDA_VISIBLE_DEVICES=3 python -m lsh.train_fmaps dataset.name="msnbc294_3" dataset.rel_mode="fhash" hashing.m_use=10  dataset.embed_dim=294 hashing.T=3  fmap_training.tr_fmap_dim=10 fmap_training.hidden_layers=[]  hashing.FUNC="sighinge" hashing.name="Fhash_Trained" training.patience=50 fmap_training.tr_fmap_loss="BCE3"  fmap_training.margin=0.05
# CUDA_VISIBLE_DEVICES=3 python -m lsh.train_fmaps dataset.name="msnbc294_4" dataset.rel_mode="fhash" hashing.m_use=10  dataset.embed_dim=294 hashing.T=3  fmap_training.tr_fmap_dim=10 fmap_training.hidden_layers=[]  hashing.FUNC="sighinge" hashing.name="Fhash_Trained" training.patience=50 fmap_training.tr_fmap_loss="BCE3"  fmap_training.margin=0.05





