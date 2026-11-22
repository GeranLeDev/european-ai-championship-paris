"""Zero-shot CLIP naming of the forget-set pseudo-classes against several vocabularies.
Usage: PYTHONPATH=<pkgs with open_clip> python clip_names.py <out_dir>"""
import sys, json, numpy as np, torch, open_clip
from pathlib import Path
from open_clip.zero_shot_metadata import IMAGENET_CLASSNAMES
root = Path(__file__).resolve().parents[2]; out = Path(sys.argv[1])
torch.set_num_threads(4)
CIFAR100 = [c for c in "apple aquarium fish baby bear beaver bed bee beetle bicycle bottle bowl boy bridge bus butterfly camel can castle caterpillar cattle chair chimpanzee clock cloud cockroach couch crab crocodile cup dinosaur dolphin elephant flatfish forest fox girl hamster house kangaroo keyboard lamp lawn mower leopard lion lizard lobster man maple tree motorcycle mountain mouse mushroom oak tree orange orchid otter palm tree pear pickup truck pine tree plain plate poppy porcupine possum rabbit raccoon ray road rocket rose sea seal shark shrew skunk skyscraper snail snake spider squirrel streetcar sunflower sweet pepper table tank telephone television tiger tractor train trout tulip turtle wardrobe whale willow tree wolf woman worm".replace("aquarium fish","aquarium_fish").replace("lawn mower","lawn_mower").replace("maple tree","maple_tree").replace("oak tree","oak_tree").replace("palm tree","palm_tree").replace("pickup truck","pickup_truck").replace("pine tree","pine_tree").replace("sweet pepper","sweet_pepper").replace("willow tree","willow_tree").split(" ")]
CIFAR100 = [c.replace("_", " ") for c in CIFAR100]
CALTECH = [c.replace("_", " ") for c in "accordion airplane anchor ant barrel bass beaver binocular bonsai brain brontosaurus buddha butterfly camera cannon car ceiling_fan cellphone chair chandelier cougar crab crayfish crocodile cup dalmatian dollar_bill dolphin dragonfly electric_guitar elephant emu euphonium ewer ferry flamingo garfield gerenuk gramophone grand_piano hawksbill headphone hedgehog helicopter ibis inline_skate joshua_tree kangaroo ketch lamp laptop llama lobster lotus mandolin mayfly menorah metronome minaret motorbike nautilus octopus okapi pagoda panda pigeon pizza platypus pyramid revolver rhino rooster saxophone schooner scissors scorpion sea_horse snoopy soccer_ball stapler starfish stegosaurus stop_sign strawberry sunflower tick trilobite umbrella watch water_lilly wheelchair wild_cat windsor_chair wrench yin_yang face leopard".split(" ")]
FREE = """fox figurine|owl figurine|rabbit figurine|heron figurine|crane bird figurine|stork figurine|turtle figurine|cat figurine|dog figurine|frog figurine|elephant figurine|horse figurine|bear figurine|deer figurine|duck figurine|
paper lantern|oil lantern|kerosene lantern|hurricane lantern|hanging lantern|wall lantern|candle lantern|stone lantern|garden lantern|street lamp|lamp post|porch light|camping lantern|railroad lantern|
glass teapot|ceramic teapot|cast iron teapot|kettle|samovar|coffee pot|teacup|pitcher|vase|
mushroom|toadstool|morel|fern|moss|pine cone|acorn|
aloe vera|agave|yucca|cactus|succulent|bonsai|potted plant|bromeliad|
suspension bridge|rope bridge|truss bridge|arch bridge|covered bridge|stone bridge|footbridge|drawbridge|viaduct|pier|
lighthouse|water tower|clock tower|bell tower|silo|
sailboat|rowboat|canoe|kayak|fishing boat|tugboat|dinghy|gondola|motorboat|yacht|ferry|raft|buoy|
wicker chair|rattan chair|folding chair|deck chair|lawn chair|camping chair|armchair|wingback chair|rocking chair|office chair|stool|bench|hammock|sofa|
dutch windmill|windmill|wind turbine|watermill|water wheel|wind pump|
knitted scarf|knitted blanket|throw blanket|shawl|sweater|woven basket|wicker basket|laundry basket|pillow|cushion|embroidered pillow|quilt|rug|ball of yarn|
leather boots|cowboy boots|combat boots|sneakers|canvas sneakers|clogs|wooden clogs|sandals|slippers|loafers|high heels|rain boots|
loaf of bread|baguette|sourdough bread|croissant|pretzel|bagel|cake|pie|cheese|
hourglass|sundial|pocket watch|wall clock|alarm clock|compass|globe|telescope|
kite|hot air balloon|paper airplane|weather vane|flag|
totem pole|pagoda|gazebo|stone statue|obelisk|torii gate|temple|pavilion|
beehive|birdhouse|bird feeder|bird nest|doghouse|mailbox|wishing well|
origami crane|origami|paper boat|anchor|ship wheel|rope|bell|lantern|candle|bicycle|typewriter|camera|umbrella|guitar|violin|book|key|ladder|barrel|wheelbarrow|watering can|scarecrow|fountain|treehouse|tent|campfire"""
FREE = [w.strip() for w in FREE.replace("\n", "").split("|") if w.strip()]
vocabs = {"imagenet1k": list(IMAGENET_CLASSNAMES), "cifar100": CIFAR100, "caltech101": CALTECH, "free": FREE}

X = np.load(root / "forget_images.npz")["images"]
y = np.load(out / "pseudo.npy")
model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
tok = open_clip.get_tokenizer("ViT-B-32"); model.eval()
mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1); std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
fp = out / "clip_b32_laion2b_img.npy"
if fp.exists(): F = torch.from_numpy(np.load(fp))
else:
    fs = []
    with torch.no_grad():
        for i in range(0, len(X), 100):
            x = torch.from_numpy(X[i:i + 100]).permute(0, 3, 1, 2).float() / 255
            fs.append(torch.nn.functional.normalize(model.encode_image((x - mean) / std), dim=-1))
    F = torch.cat(fs); np.save(fp, F.numpy())
res = {}
for name, voc in vocabs.items():
    with torch.no_grad():
        T = torch.stack([torch.nn.functional.normalize(model.encode_text(tok([t.format(w) for w in voc])), dim=-1)
                         for t in ("a photo of a {}.", "a {}.")]).mean(0)
        T = torch.nn.functional.normalize(T, dim=-1)
    P = (100 * F @ T.T).softmax(-1)
    np.save(out / f"clip_probs_{name}.npy", P.numpy().astype(np.float16))
    res[name] = {}
    for c in np.unique(y):
        m = P[y == c]
        top = m.mean(0).topk(4)
        votes = np.bincount(m.argmax(1).numpy(), minlength=len(voc))
        res[name][int(c)] = dict(n=int((y == c).sum()), top=[(voc[i], round(float(v), 3)) for v, i in zip(top.values, top.indices)],
                                 vote=(voc[int(votes.argmax())], round(float(votes.max() / len(m)), 2)))
json.dump(res, open(out / "clip_names.json", "w"), indent=1)
for c in np.unique(y):
    print(f"c{c:3d} n={res['free'][int(c)]['n']:3d} | FREE {res['free'][int(c)]['top'][:3]} vote={res['free'][int(c)]['vote']} | IN1k {res['imagenet1k'][int(c)]['top'][:2]} | C100 {res['cifar100'][int(c)]['vote']} | CT101 {res['caltech101'][int(c)]['vote']}")
for name in vocabs:
    print(name, "mean top-vote share", round(float(np.mean([v['vote'][1] for v in res[name].values()])), 3), "mean max mean-prob", round(float(np.mean([v['top'][0][1] for v in res[name].values()])), 3))
