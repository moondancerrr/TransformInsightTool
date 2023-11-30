import numpy as np
import pandas as pd
import sys
import transformers
import torch

# Local imports.
import iterable
#from train_bigbro import BigBroForTokenClassification 
from bigbro import BigBroForTokenClassification 


species = {'galdieria_sulphuraria': 0, 'vitis_vinifera': 1, 'chlamydomonas_reinhardtii': 2, 'corchorus_capsularis': 3, 'triticum_aestivum_mattis': 4, 'physcomitrium_patens': 5, 'theobroma_cacao': 6, 'eutrema_salsugineum': 7, 'populus_trichocarpa': 8, 'chara_braunii': 9, 'amborella_trichopoda': 10, 'oryza_glaberrima': 11, 'panicum_hallii': 12, 'quercus_lobata': 13, 'sesamum_indicum': 14, 'setaria_viridis': 15, 'cannabis_sativa_female': 16, 'solanum_tuberosum_rh8903916': 17, 'arabidopsis_lyrata': 18, 'trifolium_pratense': 19, 'triticum_aestivum_landmark': 20, 'oryza_rufipogon': 21, 'brassica_rapa': 22, 'prunus_avium': 23, 'arabidopsis_halleri': 24, 'oryza_brachyantha': 25, 'leersia_perrieri': 26, 'actinidia_chinensis': 27, 'triticum_dicoccoides': 28, 'brassica_rapa_ro18': 29, 'oryza_indica': 30, 'marchantia_polymorpha': 31, 'corylus_avellana': 32, 'triticum_aestivum_cadenza': 33, 'triticum_aestivum_robigus': 34, 'setaria_italica': 35, 'triticum_aestivum_norin61': 36, 'oryza_nivara': 37, 'vigna_radiata': 38, 'triticum_aestivum_stanley': 39, 'triticum_aestivum_julius': 40, 'zea_mays': 41, 'prunus_persica': 42, 'juglans_regia': 43, 'triticum_aestivum': 44, 'kalanchoe_fedtschenkoi': 45, 'triticum_aestivum_arinalrfor': 46, 'lactuca_sativa': 47, 'ostreococcus_lucimarinus': 48, 'ananas_comosus': 49, 'theobroma_cacao_criollo': 50, 'oryza_meridionalis': 51, 'arabis_alpina': 52, 'cucumis_sativus': 53, 'oryza_glumipatula': 54, 'prunus_dulcis': 55, 'solanum_lycopersicum': 56, 'nymphaea_colorata': 57, 'selaginella_moellendorffii': 58, 'capsicum_annuum': 59, 'gossypium_raimondii': 60, 'triticum_aestivum_mace': 61, 'triticum_turgidum': 62, 'camelina_sativa': 63, 'olea_europaea_sylvestris': 64, 'medicago_truncatula': 65, 'olea_europaea': 66, 'dioscorea_rotundata': 67, 'citrus_clementina': 68, 'triticum_aestivum_paragon': 69, 'cynara_cardunculus': 70, 'helianthus_annuus': 71, 'musa_acuminata': 72, 'hordeum_vulgare': 73, 'brassica_oleracea': 74, 'aegilops_tauschii': 75, 'triticum_aestivum_jagger': 76, 'oryza_sativa': 77, 'eucalyptus_grandis': 78, 'triticum_spelta': 79, 'chenopodium_quinoa': 80, 'ipomoea_triloba': 81, 'malus_domestica_golden': 82, 'triticum_urartu': 83, 'brachypodium_distachyon': 84, 'triticum_aestivum_claire': 85, 'beta_vulgaris': 86, 'eragrostis_tef': 87, 'triticum_aestivum_weebil': 88, 'arabidopsis_thaliana': 89, 'brassica_napus': 90, 'coffea_canephora': 91, 'oryza_barthii': 92, 'panicum_hallii_fil2': 93, 'oryza_punctata': 94, 'asparagus_officinalis': 95, 'solanum_tuberosum': 96, 'chondrus_crispus': 97, 'cucumis_melo': 98, 'vigna_angularis': 99, 'sorghum_bicolor': 100, 'rosa_chinensis': 101, 'pistacia_vera': 102, 'lupinus_angustifolius': 103, 'saccharum_spontaneum': 104, 'daucus_carota': 105, 'Nymphaea_colorata': 106, 'nicotiana_attenuata': 107, 'cyanidioschyzon_merolae': 108, 'triticum_aestivum_lancer': 109, 'manihot_esculenta': 110, 'eragrostis_curvula': 111, 'ficus_carica': 112, 'citrullus_lanatus': 113, 'glycine_max': 114}


#species = {'actinidia_chinensis': 0, 'aegilops_tauschii': 1, 'amborella_trichopoda': 2, 'ananas_comosus': 3, 'arabidopsis_halleri': 4, 'arabidopsis_lyrata': 5, 'arabidopsis_thaliana': 6, 'arabis_alpina': 7, 'asparagus_officinalis': 8, 'beta_vulgaris': 9, 'brachypodium_distachyon': 10, 'brassica_napus': 11, 'brassica_oleracea': 12, 'brassica_rapa': 13, 'brassica_rapa_ro18': 14, 'camelina_sativa': 15, 'cannabis_sativa_female': 16, 'capsicum_annuum': 17, 'chara_braunii': 18, 'checkpoints': 19, 'chenopodium_quinoa': 20, 'chlamydomonas_reinhardtii': 21, 'chondrus_crispus': 22, 'citrullus_lanatus': 23, 'citrus_clementina': 24, 'coffea_canephora': 25, 'corchorus_capsularis': 26, 'corylus_avellana': 27, 'cucumis_melo': 28, 'cucumis_sativus': 29, 'cyanidioschyzon_merolae': 30, 'cynara_cardunculus': 31, 'daucus_carota': 32, 'dioscorea_rotundata': 33, 'eragrostis_curvula': 34, 'eragrostis_tef': 35, 'eucalyptus_grandis': 36, 'eutrema_salsugineum': 37, 'ficus_carica': 38, 'galdieria_sulphuraria': 39, 'glycine_max': 40, 'gossypium_raimondii': 41, 'helianthus_annuus': 42, 'hordeum_vulgare': 43, 'ipomoea_triloba': 44, 'juglans_regia': 45, 'kalanchoe_fedtschenkoi': 46, 'lactuca_sativa': 47, 'leersia_perrieri': 48, 'lupinus_angustifolius': 49, 'malus_domestica_golden': 50, 'manihot_esculenta': 51, 'marchantia_polymorpha': 52, 'medicago_truncatula': 53, 'musa_acuminata': 54, 'nicotiana_attenuata': 55, 'nymphaea_colorata': 56, 'Nymphaea_colorata': 57, 'olea_europaea': 58, 'olea_europaea_sylvestris': 59, 'oryza_barthii': 60, 'oryza_brachyantha': 61, 'oryza_glaberrima': 62, 'oryza_glumipatula': 63, 'oryza_indica': 64, 'oryza_longistaminata': 65, 'oryza_meridionalis': 66, 'oryza_nivara': 67, 'oryza_punctata': 68, 'oryza_rufipogon': 69, 'oryza_sativa': 70, 'ostreococcus_lucimarinus': 71, 'other_organisms': 72, 'panicum_hallii': 73, 'panicum_hallii_fil2': 74, 'physcomitrium_patens': 75, 'pistacia_vera': 76, 'populus_trichocarpa': 77, 'prunus_avium': 78, 'prunus_dulcis': 79, 'prunus_persica': 80, 'quercus_lobata': 81, 'rosa_chinensis': 82, 'saccharum_spontaneum': 83, 'selaginella_moellendorffii': 84, 'sesamum_indicum': 85, 'setaria_italica': 86, 'setaria_viridis': 87, 'solanum_lycopersicum': 88, 'solanum_tuberosum': 89, 'solanum_tuberosum_rh8903916': 90, 'sorghum_bicolor': 91, 'theobroma_cacao': 92, 'theobroma_cacao_criollo': 93, 'trifolium_pratense': 94, 'triticum_aestivum': 95, 'triticum_aestivum_arinalrfor': 96, 'triticum_aestivum_cadenza': 97, 'triticum_aestivum_claire': 98, 'triticum_aestivum_jagger': 99, 'triticum_aestivum_julius': 100, 'triticum_aestivum_lancer': 101, 'triticum_aestivum_landmark': 102, 'triticum_aestivum_mace': 103, 'triticum_aestivum_mattis': 104, 'triticum_aestivum_norin61': 105, 'triticum_aestivum_paragon': 106, 'triticum_aestivum_robigus': 107, 'triticum_aestivum_stanley': 108, 'triticum_aestivum_weebil': 109, 'triticum_dicoccoides': 110, 'triticum_spelta': 111, 'triticum_turgidum': 112, 'triticum_urartu': 113, 'vigna_angularis': 114, 'vigna_radiata': 115, 'vitis_vinifera': 116, 'zea_mays': 117}

"""
speciess = {
    "arabidopsis_thaliana": 0,
    "brassica_napus": 1,
    "brassica_oleracea": 2,
    "arabidopsis_halleri": 3,
}


speciess = {
    "physcomitrium_patens": 0,
    "arabidopsis_thaliana": 1,
    "brassica_rapa": 2,
    "gossypium_raimondii": 3,
    "citrus_clementina": 4,
    "manihot_esculenta": 5,
    "populus_trichocarpa": 6,
    "glycine_max": 7,
    "medicago_truncatula": 8,
    "malus_domestica_golden": 9,
    "vitis_vinifera": 10,
    "zea_mays": 11,
    "amborella_trichopoda": 12,
}
"""
#speciess = { 
#    "arabidopsis_thaliana": 0,
#    "brassica_napus": 1,
#    "brassica_oleracea": 2,
#   "arabidopsis_halleri": 3,
#}

class TokenizerCollator:
   def __init__(self, tokenizer, species, max_len=8192):
      self.tokenizer = tokenizer
      self.species = species
      self.max_len = max_len

   def __call__(self, examples):
      tokenized = tokenizer(
            [" ".join(ex["seq"].upper()) for ex in examples],
            return_attention_mask=True,
            return_token_type_ids=True,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
      )

      L = tokenized["input_ids"].shape[-1]
      species_index = torch.tensor([[self.species.get(ex["species"])] for ex in examples])
      tokenized["token_type_ids"] = species_index.repeat(1, L)

      tokenized["labels"] = torch.tensor([
          [-100] + ex["calls"][:L-2] + [-100] + [-100]*(L-2-len(ex["calls"])) for ex in examples
      ])
      tokenized["id"] = examples[0]["id"]
      return tokenized

class Collator:
   def __init__(self, tokenizer):
      self.tokenizer = tokenizer

   def __call__(self, examples):
      tokenized = tokenizer(
            [" ".join(ex["seq"].upper()) for ex in examples],
            return_attention_mask=True,
            return_token_type_ids=False,
            padding=True,
            return_tensors="pt"
      )
      tokenized["id"] = [ex["id"] for ex in examples]
      tokenized["calls"] = [ex["calls"] for ex in examples]
      return tokenized


if __name__ == "__main__":
   tokenizer = transformers.PreTrainedTokenizerFast(
         tokenizer_file="./TokenizerModel/model.json",
         bos_token="[CLS]",
         eos_token="[SEP]",
         unk_token="[UNK]",
         sep_token="[SEP]",
         pad_token="[PAD]",
         cls_token="[CLS]",
         mask_token="[MASK]"
   )

   config = transformers.BigBirdConfig(vocab_size=len(tokenizer),
         attention_type="block_sparse",
         max_position_embeddings=8192, 
         sep_token_id=2,
         type_vocab_size=115,
         embedding_size=768, rotary_value=False)
   model = BigBroForTokenClassification(config=config)
   sd = torch.load("my_third_checkpoint.pt-v1.ckpt")['state_dict']
   model.load_state_dict({ k.replace("model.",""):v for (k,v) in sd.items()}, strict = True)
   model.bert.set_attention_type("block_sparse")

   data = iterable.IterableJSONData("species.json")
   junctions = {}
   with open('GT_shuf.txt') as f:
    for line in f:
        x, junction, gene_id, position, rank = line.strip().split()
        key = f"{gene_id}_{junction}_{rank}_{position}"
        junctions[key] = x.strip('>')


   from torch.utils.data import DataLoader
   dataloader = DataLoader(
         data,
         shuffle=False,
         collate_fn = TokenizerCollator(tokenizer, species),
         batch_size=1
   )

   model.eval()
   #model.to("cuda")
   model.to("cuda:1")
   with open("GT.vectors_final.txt", "w") as f:

      for x, batch in enumerate(dataloader):
         labels = batch.pop("labels")
         gene_id = batch.pop("id")
         for key in batch:
            batch[key] = batch[key].to("cuda:1")
         with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True)
            a = [i.split('_') for i in junctions.keys() if i.split('_')[0]==gene_id]
            for i in a:
               last_hidden_state_GT = outputs.last_hidden_state[0,int(i[3])+1:int(i[3])+3,:].view(1, -1)
               last_hidden_state_junction = outputs.last_hidden_state[0,int(i[3])+1:int(i[3])+3,:].view(1, -1)
               logit = torch.softmax(outputs.logits, dim=-1).cpu()
               prob = torch.mean(logit[:,int(i[3])+1:int(i[3])+3,1])
               df_GT = pd.DataFrame(last_hidden_state_GT.cpu().view(1, -1).numpy())
               df_junction = pd.DataFrame(last_hidden_state_junction.cpu().view(1, -1).numpy())
               df_junction['junction'] = '_'.join([j for j in i])
               df_GT['junction'] = '_'.join([j for j in i])
               df_GT.set_index('junction', inplace=True)
               df_junction.set_index('junction', inplace=True)
               df = pd.concat([df_GT,df_junction],axis=1)
               df['logits_junction'] = (torch.mean(logit[:,int(i[3]):int(i[3])+2,1])).numpy()
               df['logits_GT'] = prob.numpy()
               df.to_csv(f, header=False, sep=' ')

