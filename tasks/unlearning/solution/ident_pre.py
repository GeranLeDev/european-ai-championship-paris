"""Is orig a fine-tune of a public timm ViT-B/16? cosine similarity of weights with candidate pretrained checkpoints."""
import sys, torch, timm
sd = torch.load("../model.pt", weights_only=True)
for name in sys.argv[1:]:
    try:
        p = timm.create_model(name, pretrained=True).state_dict()
    except Exception as e:
        print(name, "ERR", str(e)[:80]); continue
    out = []
    for k in ["patch_embed.proj.weight", "pos_embed", "cls_token", "blocks.0.attn.qkv.weight", "blocks.5.mlp.fc1.weight", "blocks.11.mlp.fc2.weight", "norm.weight"]:
        if k in p and p[k].shape == sd[k].shape:
            out.append(f"{k.split('.')[0]}.{k.split('.')[-2] if '.' in k else ''}={torch.nn.functional.cosine_similarity(sd[k].flatten().float(), p[k].flatten().float(), dim=0):.3f}")
    print(name, " ".join(out), flush=True)
