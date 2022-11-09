elif self.embedding_selector:
# pdb.set_trace()
if self.use_rl:
    if self.use_embedding_masks:
        sentence_tensor = [sentences.features[x].to(flair.device) for idx, x in
                           enumerate(sorted(sentences.features.keys()))]
        sentence_masks = [
            torch.ones_like(sentence_tensor[idx]) * sentences.embedding_mask[:, idx, None, None].to(flair.device) for
            idx, x in enumerate(sorted(sentences.features.keys()))]
        sentence_tensor = torch.cat([x * sentence_masks[idx] for idx, x in enumerate(sentence_tensor)], -1)
    else:
        if self.embedding_attention:
            embatt = torch.sigmoid(self.selector)
            sentence_tensor = torch.cat(
                [sentences.features[x].to(flair.device) * self.selection[idx] * embatt[idx] for idx, x in
                 enumerate(sorted(sentences.features.keys()))], -1)
        else:
            sentence_tensor = torch.cat([sentences.features[x].to(flair.device) * self.selection[idx] for idx, x in
                                         enumerate(sorted(sentences.features.keys()))], -1)
# sentence_tensor = torch.cat([sentences.features[x].to(flair.device) * self.selection[idx] for idx, x in enumerate(sentences.features.keys())],-1)
else:
    if self.use_gumbel:
        if self.training:
            selection = torch.nn.functional.gumbel_softmax(self.selector, hard=True)
            sentence_tensor = torch.cat([sentences.features[x].to(flair.device) * selection[idx][1] for idx, x in
                                         enumerate(sorted(sentences.features.keys()))], -1)
        else:
            selection = torch.argmax(self.selector, -1)
            sentence_tensor = torch.cat([sentences.features[x].to(flair.device) * selection[idx] for idx, x in
                                         enumerate(sorted(sentences.features.keys()))], -1)
    else:
        selection = torch.sigmoid(self.selector)
        sentence_tensor = torch.cat([sentences.features[x].to(flair.device) * selection[idx] for idx, x in
                                     enumerate(sorted(sentences.features.keys()))], -1)
