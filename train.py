import os
import time
import json
from tensorboardX import SummaryWriter

from validate import validate, validate_multiclass
from data import create_dataloader
from earlystop import EarlyStopping
from networks.trainer import Trainer
from options.train_options import TrainOptions


"""Currently assumes jpg_prob, blur_prob 0 or 1"""
def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.data_label = 'val'
    val_opt.jpg_method = ['pil']
    if val_opt.data_mode == 'csv_multiclass':
        val_opt.blur_prob = 0
        val_opt.jpg_prob = 0
        val_opt.no_flip = True
    if len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_opt.jpg_qual) != 1:
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]

    return val_opt



if __name__ == '__main__':
    opt = TrainOptions().parse()
    val_opt = get_val_opt()
 
    model = Trainer(opt)
    
    data_loader = create_dataloader(opt)
    val_loader = create_dataloader(val_opt)

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))
        
    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.001, verbose=True)
    start_time = time.time()
    print ("Length of data loader: %d" %(len(data_loader)))
    for epoch in range(opt.niter):
        
        for i, data in enumerate(data_loader):
            model.total_steps += 1

            model.set_input(data)
            model.optimize_parameters()

            if model.total_steps % opt.loss_freq == 0:
                print("Train loss: {} at step: {}".format(model.loss, model.total_steps))
                train_writer.add_scalar('loss', model.loss, model.total_steps)
                print("Iter time: ", ((time.time()-start_time)/model.total_steps)  )

            if model.total_steps in [10,30,50,100,1000,5000,10000] and False: # save models at these iters 
                model.save_networks('model_iters_%s.pth' % model.total_steps)

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d' % (epoch))
            model.save_networks( 'model_epoch_latest.pth' )
            model.save_networks( 'model_epoch_%s.pth' % epoch )

        # Validation
        model.eval()
        if opt.num_classes == 1:
            ap, r_acc, f_acc, acc = validate(model.model, val_loader)
            val_writer.add_scalar('accuracy', acc, model.total_steps)
            val_writer.add_scalar('ap', ap, model.total_steps)
            print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))
            validation_score = acc
        else:
            metrics = validate_multiclass(model.model, val_loader, opt.class_names)
            validation_score = metrics["macro_f1"]
            scalar_keys = [
                "accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "weighted_precision",
                "weighted_recall",
                "weighted_f1",
            ]
            for key in scalar_keys:
                val_writer.add_scalar(key, metrics[key], model.total_steps)

            metrics_record = {
                "epoch": epoch,
                "total_steps": model.total_steps,
                **metrics,
            }
            metrics_path = os.path.join(opt.checkpoints_dir, opt.name, "val_metrics.jsonl")
            with open(metrics_path, "a") as f:
                f.write(json.dumps(metrics_record) + "\n")

            print(
                "(Val @ epoch {}) macro_f1: {:.4f}; accuracy: {:.4f}; "
                "weighted_f1: {:.4f}; macro_precision: {:.4f}; macro_recall: {:.4f}".format(
                    epoch,
                    metrics["macro_f1"],
                    metrics["accuracy"],
                    metrics["weighted_f1"],
                    metrics["macro_precision"],
                    metrics["macro_recall"],
                )
            )

        early_stopping(validation_score, model)
        if early_stopping.early_stop:
            cont_train = model.adjust_learning_rate()
            if cont_train:
                print("Learning rate dropped by 10, continue training...")
                early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.002, verbose=True)
            else:
                print("Early stopping.")
                break
        model.train()
