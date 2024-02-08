import torch
import torch.nn as nn

class trainer:
    def __init__():
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train(self, train_loader, epochs):
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for image, audio in train_loader:
                image, audio = image.to(self.device), audio.to(self.device)
                # optimizer
                self.optimizer.ADAMW
                outputs = self.model(image, audio)
                loss = self.criterion(outputs)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            epoch_loss = running_loss/ len(train_loader.AudioSet)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
    
    def build_train_loader():
        
        return train_loader
    
    def build_lr_scheduler():

        return lr_schedular
         
    def build_optimizer():

        return optimizer
             
"""
def setup(args):

    cfg = get_cfg()

    return cfg
    
def main(args):
    cfg = setup(args)

    return trainer.train()

if __name__ == "__main__":
    args = 
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
"""      