train.py中目前有out_plug和out_base

先运行out_base，并且使用with torch.no_grad():包裹，然后再运行out_plug

vision_embeds目前(2026/2/28)还是torch.zeros占位