if self.map_embeddings:
    new_list = []
    # pdb.set_trace()
    for idx, embedding_name in enumerate(sorted(sentences.features.keys())):
        new_list.append(self.map_linears[idx](sentences.features[embedding_name].to(flair.device)))
    sentence_tensor = torch.cat(new_list, -1)