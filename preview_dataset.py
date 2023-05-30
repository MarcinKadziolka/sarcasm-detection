from headline_data_set import HeadlineDataset
import torch

filter_h = [4,6,8]
train_sampler = None 
batch_size = 32
workers = 4

train_dataset = HeadlineDataset(
    csv_file='DATA/txt/headline_train.txt', 
    word_embedding_file='DATA/embeddings/headlines_filtered_embs.txt', 
    pad = max(filter_h) - 1,
    whole_data='DATA/txt/headlines_clean.txt',
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
    num_workers=workers, pin_memory=True)


val_dataset = HeadlineDataset(
    csv_file='DATA/txt/headline_val.txt', 
    word_embedding_file='DATA/embeddings/headlines_filtered_embs.txt', 
    pad = max(filter_h) - 1,
    word_idx = train_dataset.word_idx,
    pretrained_embs = train_dataset.pretrained_embs,
    max_l=train_dataset.max_l,
)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=None,
    num_workers=workers, pin_memory=True)

test_dataset = HeadlineDataset(
    csv_file='DATA/txt/headline_test.txt', 
    word_embedding_file='DATA/embeddings/headlines_filtered_embs.txt', 
    pad = max(filter_h) - 1,
    word_idx = train_dataset.word_idx,
    pretrained_embs = train_dataset.pretrained_embs,
    max_l=train_dataset.max_l,
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=None,
    num_workers=workers, pin_memory=True)




# display train examples:
for i, (arr, label, sent) in enumerate(train_loader):
    print(arr[0].shape)
    print()
    print(label[0], sent[0])
    break

