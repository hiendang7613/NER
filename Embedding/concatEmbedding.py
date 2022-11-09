sentence_tensor = torch.cat([sentences.features[x].to(flair.device) for x in sorted(sentences.features.keys())], -1)
