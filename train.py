"""
Much of the code is taken from this blog: https://mccormickml.com/2019/07/22/BERT-fine-tuning/
"""
import random
import time

import numpy as np
import torch
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import utils
from model.WIQAClassifier import WIQAClassifier
from src.helpers.data_reader import get_wiqa_dataloader


class Trainer:

    def __init__(self, dataloader_dir):
        super().__init__()
        self.dataloader_dir = dataloader_dir
        self.n_epochs = 30
        self.n_labels = 3
        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.init_model()
        self.init_dataloaders()
        self.init_optimizer()
        self.init_scheduler()

    def init_model(self):
        model = WIQAClassifier.from_pretrained(
            "bert-base-uncased",
            num_labels=self.n_labels,
            output_attentions=False,
            output_hidden_states=False
        )
        model.cuda()
        self.model = model

    def init_optimizer(self):
        self.optimizer = AdamW(self.model.parameters(),
                               lr=2e-5,
                               eps=1e-8
                               )

    def init_dataloaders(self):
        self.train_dataloader = get_wiqa_dataloader(self.dataloader_dir, "train")
        self.dev_dataloader = get_wiqa_dataloader(self.dataloader_dir, "dev")
        self.test_dataloader = get_wiqa_dataloader(self.dataloader_dir, "test")

    def init_scheduler(self):
        total_steps = len(self.train_dataloader) * self.n_epochs

        # Create the learning rate scheduler.
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=0,  # Default value in run_glue.py
                                                         num_training_steps=total_steps)

    def train(self):

        training_stats = []
        total_t0 = time.time()
        for epoch_i in range(0, self.n_epochs):

            # ========================================
            #               Training
            # ========================================

            print("")
            print(
                '======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.n_epochs))
            print('Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()

            # Reset the total loss for this epoch.
            total_train_loss = 0

            self.model.train()

            # For each batch of training data...
            for step, batch in enumerate(self.train_dataloader):

                if step % 40 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = utils.format_time(time.time() - t0)

                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(
                        step, len(self.train_dataloader), elapsed))
                (passage_input_ids, passage_attention_mask, passage_token_type_ids,
                 labels) = batch
                """
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)
                """
                self.model.zero_grad()

                loss, logits = self.model(passage_input_ids=passage_input_ids.to(self.device),
                                          passage_attention_mask=passage_attention_mask.to(self.device),
                                          passage_token_type_ids=passage_token_type_ids.to(self.device),
                                          labels=labels.to(self.device))
                total_train_loss += loss.item()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                self.scheduler.step()

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(self.train_dataloader)

            # Measure how long this epoch took.
            training_time = utils.format_time(time.time() - t0)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(training_time))

            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

            print("")
            print("Running Validation...")

            t0 = time.time()

            self.model.eval()

            total_eval_accuracy = 0
            total_eval_loss = 0
            nb_eval_steps = 0

            # Evaluate data for one epoch
            for batch in self.dev_dataloader:
                (passage_input_ids, passage_attention_mask, passage_token_type_ids,
                 labels) = batch
    
                with torch.no_grad():
                    loss, logits = self.model(passage_input_ids=passage_input_ids.to(self.device),
                            passage_attention_mask=passage_attention_mask.to(self.device),
                            passage_token_type_ids=passage_token_type_ids.to(self.device),
                            labels=labels.to(self.device))
         

                total_eval_loss += loss.item()

                logits = logits.detach().cpu().numpy()
                label_ids = labels.to('cpu').numpy()

                total_eval_accuracy += utils.flat_accuracy(logits, label_ids)

            # Report the final accuracy for this validation run.
            avg_val_accuracy = round(
                total_eval_accuracy / len(self.dev_dataloader), 4)
            print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

            # Calculate the average loss over all of the batches.
            avg_val_loss = total_eval_loss / len(self.dev_dataloader)

            # Measure how long the validation run took.
            validation_time = utils.format_time(time.time() - t0)

            print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))

            # Record all statistics from this epoch.
            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Accur.': avg_val_accuracy,
                    'Training Time': training_time,
                    'Validation Time': validation_time
                }
            )

            self.save_model(avg_val_accuracy)

        print("")
        print("Training complete!")

        print("Total training took {:} (h:mm:ss)".format(
            utils.format_time(time.time()-total_t0)))


    def save_model(self, tag):
        tag = str(tag)
        import os
        output_dir = f'./{self.dataloader_dir}/models/checkpoint-{tag}'

        # Create output directory if needed
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("Saving model to %s" % output_dir)

        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        model_to_save = self.model.module if hasattr(
            self.model, 'module') else self.model
        model_to_save.save_pretrained(output_dir)


if __name__ == '__main__':
    import sys
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    trainer = Trainer(dataloader_dir=sys.argv[1])
    trainer.save_model("inf-loss-based")
    trainer.train()
