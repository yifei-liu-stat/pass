import torch
from torch.utils.data import Dataset, DataLoader


class IMDBdataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = self.reviews[idx]
        target = self.targets[idx]

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].flatten()
        attention_mask = encoding["attention_mask"].flatten()

        return {
            "review_text": review,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "targets": torch.tensor(target, dtype=torch.long),
        }


def create_data_loader_imdb(df, tokenizer, max_len, batch_size):
    """Create dataloader from IMDB movie review dataset"""
    dataset = IMDBdataset(
        reviews=df.review.to_numpy(),
        targets=df.sentiment.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len,
    )

    return DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)
