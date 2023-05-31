from torch import nn
from transformer import CTransformer
from headline_data_set import HeadlineDataset
import torch
from sklearn.metrics import accuracy_score
import tqdm
import wandb
import os


def train(model, train_loader, val_loader, num_epochs, criterion, optimizer):
    epoch_progress = tqdm.tqdm(range(num_epochs))
    for epoch in epoch_progress:
        model.train()
        train_loss = 0
        y_true = []
        y_pred = []
        for i, (input, target, sent) in enumerate(train_loader):
            input = input.to(device)
            target = target.to(device).float()

            optimizer.zero_grad()
            output = model(input)

            loss = criterion(output, target)
            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            y_true.extend(target.tolist())
            y_pred.extend(torch.where(output > 0.5, 1, 0).tolist())

        
        model.eval()
        val_loss = 0
        val_y_true = []
        val_y_pred = []
        with torch.no_grad():
            for val_input, val_target, _ in val_loader:
                val_input = val_input.to(device)
                val_target = val_target.to(device).float()

                val_output = model(val_input)

                loss = criterion(val_output, val_target)
                val_loss += loss.item()

                val_y_true.extend(val_target.tolist())
                val_y_pred.extend(torch.where(val_output > 0.5, 1, 0).tolist())

        accuracy = accuracy_score(y_true, y_pred)
        val_accuracy = accuracy_score(val_y_true, val_y_pred)

        epoch_progress.set_description(f"Epoch: [{epoch+1}/{num_epochs}], train loss: {train_loss:.4f}, train accuracy: {accuracy:.4f}, val loss: {val_loss:.4f}, val accuracy: {val_accuracy:.4f}")
        wandb.log({"train_loss": train_loss, "train_accuracy": accuracy, "val_loss": val_loss, "val_accuracy": val_accuracy})

def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    test_y_true = []
    test_y_pred = []
    with torch.no_grad():
        for test_input, test_target, _ in test_loader:
            test_input = test_input.to(device)
            test_target = test_target.to(device).float()
            test_output = model(test_input)

            loss = criterion(test_output, test_target)
            test_loss += loss.item()

            test_y_true.extend(test_target.tolist())
            test_y_pred.extend(torch.where(test_output > 0.5, 1, 0).tolist())

    test_accuracy = accuracy_score(test_y_true, test_y_pred)

    print(f"Test loss: {test_loss:.4f}, test accuracy: {test_accuracy:.4f}")
    wandb.log({"test_loss": test_loss, "test_accuracy": test_accuracy})

# To disable wandb change use mode="disabled"
wandb.init(project="sarcasm-detection", config='config.yaml', mode="disabled")
config = wandb.config
print("---------------------------------------")
print(f"Arguments received: ")
print("---------------------------------------")
for k, v in sorted(config.items()):
    print(f"{k:25}: {v}")
print("---------------------------------------")

# Don't change
filter_h = [4,6,8]
train_sampler = None

train_dataset = HeadlineDataset(
    csv_file='DATA/txt/headline_train.txt', 
    word_embedding_file='DATA/embeddings/headlines_filtered_embs.txt', 
    pad = max(filter_h) - 1,
    whole_data='DATA/txt/headlines_clean.txt',
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=config.batch_size, shuffle=(train_sampler is None),
    num_workers=config.num_workers, pin_memory=True)


val_dataset = HeadlineDataset(
    csv_file='DATA/txt/headline_val.txt', 
    word_embedding_file='DATA/embeddings/headlines_filtered_embs.txt', 
    pad = max(filter_h) - 1,
    word_idx = train_dataset.word_idx,
    pretrained_embs = train_dataset.pretrained_embs,
    max_l=train_dataset.max_l,
)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=config.batch_size, shuffle=None,
    num_workers=config.num_workers, pin_memory=True)

test_dataset = HeadlineDataset(
    csv_file='DATA/txt/headline_test.txt', 
    word_embedding_file='DATA/embeddings/headlines_filtered_embs.txt', 
    pad = max(filter_h) - 1,
    word_idx = train_dataset.word_idx,
    pretrained_embs = train_dataset.pretrained_embs,
    max_l=train_dataset.max_l,
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=config.batch_size, shuffle=None,
    num_workers=config.num_workers, pin_memory=True)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Val dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")


num_tokens = train_dataset.pretrained_embs.shape[0]
seq_length = 100
print(f"Number of tokens: {num_tokens}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = CTransformer(
        emb=config.embedding_dim, 
        heads=config.num_heads, 
        depth=config.depth, 
        seq_length=seq_length, 
        num_tokens=num_tokens, 
        max_pool=True, 
        dropout=config.dropout,
        attention_type=config.attention_type
        ).to(device)

print(f"Model is on {next(model.parameters()).device}")

if config.optimizer == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr = config.lr,
                                 betas=(config.momentum, config.beta2),
                                 weight_decay=config.weight_decay)
elif config.optimizer == "AdamW":
    optimizer = torch.optim.AdamW(model.parameters(), lr = config.lr,
                                  betas=(config.momentum, config.beta2),
                                  weight_decay=config.weight_decay)
elif config.optimizer == "Adadelta":
    optimizer = torch.optim.Adadelta(model.parameters(), lr = config.lr,
                                        rho=config.momentum, 
                                         weight_decay=config.weight_decay)
elif config.optimizer == "SGD":
    optimizer = torch.optim.SGD(model.parameters(), lr = config.lr,
                                    momentum=config.momentum, 
                                     weight_decay=config.weight_decay)
else:
    raise ValueError(f"Unknown optimizer: {config.optimizer}")



model_state_dict_name = "model_state_dict"
optimizer_state_dict_name = "optimizer_state_dict"

if config.restore_from is not None:
    checkpoint = torch.load(config.restore_from)
    model.load_state_dict(checkpoint[model_state_dict_name])
    optimizer.load_state_dict(checkpoint[optimizer_state_dict_name])
    print(f"Restored from {config.restore_from} and continuing training...")
else:
    print("Training from scratch...")

criterion = nn.BCELoss()

train(model, train_loader, val_loader, config.num_epochs, criterion, optimizer)
print("Training finished!")
print("Testing...")
test(model, test_loader, criterion)
print("Testing finished!")
print("Saving model...")
checkpoint_path = os.path.join(wandb.run.dir, "checkpoint.tar")
torch.save({
            model_state_dict_name: model.state_dict(),
            optimizer_state_dict_name: optimizer.state_dict(),
            }, checkpoint_path)
print(f"Saved to {checkpoint_path}")
