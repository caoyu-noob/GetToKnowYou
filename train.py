import json
import logging
import os
import random

import torch
from tensorboardX import SummaryWriter

from config import get_trainer_config
from config import InputConfig
from model.dataset import FacebookDataset
from model.trainer import Trainer
from model.utils import config_logger
from model.utils import f1_score
from model.utils import open
from model.utils import set_seed
from model.seq2seq import TransformerSeq2Seq
from model.seq2seq_vocab import Seq2seqVocab
import jsonlines
from new_metrics import nlp_metrics, nltk_bleu, triple_acc_f1


class DummyWriter:
    """ Used for distributed training (from NVIDIA apex example).
        A dummy logger used so that only the main process write and log informations.
    """
    def __init__(self, *input, **kwargs):
        self.log_dir = "runs/dummy_logs/"

    def add_scalar(self, *input, **kwargs):
        pass

def get_model_and_tokenizer(args, trainer_config, logger):
    seq2seq_vocab = Seq2seqVocab(trainer_config.train_datasets, trainer_config.valid_datasets,
                                 trainer_config.test_datasets, args.vocab_path, data_type=trainer_config.data_type)
    tokenizer = seq2seq_vocab.vocab
    args.dialog_embeddings = False
    if args.model_type == 'seq2seq':
        model = TransformerSeq2Seq(args.emb_dim, args.hidden_dim, args.num_layers, args.heads, args.depth_size,
                               args.filter_size, tokenizer, args.pretrained_emb_file,
                               args.pointer_gen, logger, label_smoothing=args.label_smoothing)
    else:
        model = TransformerSeq2Seq(args.emb_dim, args.hidden_dim, args.num_layers, args.heads, args.depth_size,
                                   args.filter_size, tokenizer, args.pretrained_emb_file,
                                   args.pointer_gen, logger, base_model='gru', label_smoothing=args.label_smoothing)
    return model, tokenizer

'''Modify the model to make it fit the data'''
def modify_model(args, model, tokenizer):
    model.padding_idx = tokenizer.pad_id
    model.n_pos_embeddings = 512

    model.bos_id = tokenizer.bos_id
    model.eos_id = tokenizer.eos_id
    model.beam_size = args.beam_size
    model.diversity_groups = 1
    model.max_seq_len = 32
    model.dialog_embeddings = args.dialog_embeddings
    model.bs_temperature = args.bs_temperature
    model.bs_nucleus_p = args.bs_nucleus_p
    model.annealing_topk = args.annealing_topk
    model.length_penalty_coef = args.length_penalty
    model.vocab = None
    model.annealing = args.annealing
    model.diversity_coef = args.diversity_coef
    model.sample = False
    model.inference_mode = args.inference_mode
    model.response_k = args.response_k

def training_procedure(args, trainer_config, model, tokenizer, device, writer, logger,
                       best_checkpoint_path, last_checkpoint_path, interrupt_checkpoint_path, log_dir, test_data_type=None):
    logger.info("trainer config: {}".format(trainer_config))
    logger.info('loading datasets')
    train_dataset = FacebookDataset(trainer_config.train_datasets, tokenizer,
                                    max_lengths=model.n_pos_embeddings - 1,  # A bit restrictive here
                                    dialog_embeddings=args.dialog_embeddings,
                                    cache=trainer_config.train_datasets_cache,
                                    use_start_end=args.use_start_end,
                                    augment=trainer_config.persona_augment,
                                    aug_syn_proba=trainer_config.persona_aug_syn_proba,
                                    limit_size=trainer_config.limit_train_size,
                                    max_history_size=trainer_config.max_history_size,
                                    data_type=trainer_config.data_type)
    valid_dataset = FacebookDataset(trainer_config.valid_datasets, tokenizer,
                                    max_lengths=model.n_pos_embeddings - 1,  # A bit restrictive here
                                    dialog_embeddings=args.dialog_embeddings,
                                    cache=trainer_config.valid_datasets_cache,
                                    use_start_end=args.use_start_end,
                                    augment=False,
                                    aug_syn_proba=0.0,
                                    limit_size=trainer_config.limit_eval_size,
                                    max_history_size=trainer_config.max_history_size,
                                    data_type=trainer_config.data_type)
    if test_data_type is None:
        test_data_type = trainer_config.data_type
    test_dataset = FacebookDataset(trainer_config.test_datasets, tokenizer,
                                   max_lengths=model.n_pos_embeddings - 1,  # A bit restrictive here
                                   dialog_embeddings=args.dialog_embeddings,
                                   cache=trainer_config.test_datasets_cache,
                                   use_start_end=args.use_start_end,
                                   augment=False,
                                   aug_syn_proba=0.0,
                                   limit_size=trainer_config.limit_eval_size,
                                   max_history_size=trainer_config.max_history_size,
                                   data_type=test_data_type)
    logger.info('train dataset {} valid dataset {} test dataset {}'
                .format(len(train_dataset), len(valid_dataset), len(test_dataset)))

    '''Normal training will use normal trainer'''
    model_trainer = Trainer(model,
                            train_dataset,
                            trainer_config,
                            writer,
                            logger=logger,
                            valid_dataset=valid_dataset,
                            test_dataset=test_dataset,
                            n_jobs=trainer_config.n_jobs,
                            device=device,
                            ignore_idxs=tokenizer.all_special_ids,
                            evaluate_full_sequences=trainer_config.evaluate_full_sequences,
                            full_input=trainer_config.full_input,
                            best_model_path=best_checkpoint_path,
                            pointer_gen=args.pointer_gen)

    if args.load_last:
        state_dict = torch.load(trainer_config.load_last, map_location=device)
        model_trainer.load_state_dict(state_dict)

    # helpers -----------------------------------------------------
    def external_metrics_func(full_references, full_predictions, full_predictions_given_predicate, epoch, is_best=False):
        if epoch == -1:
            if is_best:
                references_file_path = os.path.join(writer.logdir, trainer_config.test_references_file)
                predictions_file_path = os.path.join(writer.logdir,  trainer_config.test_predictions_file_best)
                predictions_file_path_predicate = os.path.join(writer.logdir, trainer_config.test_predictions_file_best
                                                               + "_predicate")
            else:
                references_file_path = os.path.join(writer.logdir, trainer_config.test_references_file)
                predictions_file_path = os.path.join(writer.logdir, trainer_config.test_predictions_file_last)
                predictions_file_path_predicate = os.path.join(writer.logdir, trainer_config.test_predictions_file_last
                                                               + "_predicate")
        else:
            references_file_path = os.path.join(writer.logdir, trainer_config.eval_references_file)
            predictions_file_path = os.path.join(writer.logdir,
                                                 trainer_config.eval_predictions_file + "_{}".format(epoch))
            predictions_file_path_predicate = os.path.join(writer.logdir,
                                                 trainer_config.eval_predictions_file + "_predicate_{}".format(epoch))
        if not os.path.exists(references_file_path):
            with jsonlines.open(references_file_path, 'w') as f:
                f.write(full_references)
        with jsonlines.open(predictions_file_path, 'w') as f:
            f.write(full_predictions)
        with jsonlines.open(predictions_file_path_predicate, 'w') as f:
            f.write(full_predictions_given_predicate)
        full_references_string = [[' '.join(r) for r in ref] for ref in full_references]
        full_predictions_string = [[' '.join(p) for p in pred] for pred in full_predictions]
        full_predictions_given_predicate_string = [[' '.join(p) for p in pred] for pred in full_predictions_given_predicate]
        pred_bleu1 = nltk_bleu(full_references_string, full_predictions_string)
        pred_predicate_bleu1 = nltk_bleu(full_references_string, full_predictions_given_predicate_string)
        pred_acc, pred_f1 = triple_acc_f1(full_references_string, full_predictions_string)
        pred_predicate_acc, pred_predicate_f1 = triple_acc_f1(full_references_string, full_predictions_given_predicate_string)
        metrics = {"pred_blue1": pred_bleu1 * 100, "pred_predicate_bleu1": pred_predicate_bleu1 * 100,
                   "pred_acc": pred_acc * 100, "pred_f1": pred_f1 * 100, "pred_predicate_acc": pred_predicate_acc * 100,
                   "pred_predicate_f1": pred_predicate_f1 * 100}
        for k, v in metrics.items():
            metrics[k] = round(v, 6)

        return metrics

    # def external_metrics_func(full_references, full_predictions, epoch, is_best=False):
    #     if epoch == -1:
    #         if is_best:
    #             references_file_path = os.path.join(writer.logdir, trainer_config.test_references_file)
    #             predictions_file_path = os.path.join(writer.logdir,  trainer_config.test_predictions_file_best)
    #         else:
    #             references_file_path = os.path.join(writer.logdir, trainer_config.test_references_file)
    #             predictions_file_path = os.path.join(writer.logdir, trainer_config.test_predictions_file_last)
    #     else:
    #         references_file_path = os.path.join(writer.logdir, trainer_config.eval_references_file)
    #         predictions_file_path = os.path.join(writer.logdir,
    #                                              trainer_config.eval_predictions_file + "_{}".format(epoch))
    #
    #     if not os.path.exists(references_file_path):
    #         with open(references_file_path, 'w', encoding='utf-8') as f:
    #             f.write('\n'.join(full_references))
    #     # print(len(full_predictions))
    #     with open(os.path.join(writer.logdir, 'tt.json'), 'w') as f:
    #         json.dump(full_predictions, f)
    #     with open(predictions_file_path, 'w', encoding='utf-8') as f:
    #         if len(full_predictions[-1]) == 0:
    #             full_predictions[-1] = 'a '
    #         f.write('\n'.join(full_predictions))
    #
    #     bleu, bleu_list, nist, nist_list, nist_bleu, nist_bleu_list, s_dist, c_dist, entropy, meteor, \
    #     rouge_l, f1_score, avg_length = nlp_metrics(references_file_path, predictions_file_path, root_path=log_dir)
    #
    #     metrics = {'meteor': meteor * 100, 'avg_len': avg_length, 'rouge-l': rouge_l * 100, 'bleu': bleu, 'nist': nist,
    #                'nist-bleu': nist_bleu, 'f1': f1_score * 100}
    #     for name, metric in (
    #     ('bleu', bleu_list), ('nist', nist_list), ('nist_bleu', nist_bleu_list), ('entropy', entropy),
    #     ('sentence_div', s_dist), ('corpus_div', c_dist)):
    #         for i, m in enumerate(metric, 1):
    #             if name == 'sentence_div' or name == 'corpus_div':
    #                 metrics['{}_{}'.format(name, i)] = m * 100
    #             else:
    #                 metrics['{}_{}'.format(name, i)] = m
    #     for k, v in metrics.items():
    #         metrics[k] = round(v, 6)
    #
    #     return metrics

    def save_func(epoch):
        if epoch != -1:
            torch.save(model_trainer.model.state_dict(), last_checkpoint_path)
            logger.info('Model on Epoch %d has been saved', epoch)

    def sample_text_func(epoch):
        n_samples = 0
        model_trainer.model.eval()
        samples_idxs = random.sample(range(len(valid_dataset)), n_samples)
        samples = [valid_dataset[idx] for idx in samples_idxs]
        for persona_info, dialog, target, _ in samples:
            contexts = [torch.tensor([c], dtype=torch.long, device=model_trainer.device) for c in [persona_info, dialog]
                        if len(c) > 0]
            prediction = model_trainer.model.predict(contexts)[0]

            persona_info_str = tokenizer.ids2string(persona_info[1:-1])
            dialog_str = tokenizer.ids2string(dialog)
            dialog_str = dialog_str.replace(tokenizer.talker1_bos, '\n\t- ').replace(tokenizer.talker2_bos, '\n\t- ')
            dialog_str = dialog_str.replace(tokenizer.talker1_eos, '').replace(tokenizer.talker2_eos, '')
            target_str = tokenizer.ids2string(target[1:-1])
            prediction_str = tokenizer.ids2string(prediction)

            logger.info('\n')
            logger.info('Persona info:\n\t{}'.format(persona_info_str))
            logger.info('Dialog:{}'.format(dialog_str))
            logger.info('Target:\n\t{}'.format(target_str))
            logger.info('Prediction:\n\t{}'.format(prediction_str))

    def test_func(epoch):
        if (epoch + 1) % trainer_config.test_period == 0:
            metric_funcs = {'f1_score': f1_score}
            model_trainer.test(metric_funcs, external_metrics_func, epoch)

    def f1_risk(predictions, targets):
        scores = f1_score(predictions, targets, average=False)
        assert all([0 <= s <= 1.0 for s in scores])
        return [1 - s for s in scores]

    def get_risk_metric_func(risk_metric):
        """ risk_metric selected in:
            f1, meteor, avg_len, nist_{1, 2, 3, 4}, entropy_{1, 2, 3, 4}, div_{1, 2}, bleu_{1, 2, 3, 4}
        """

        def external_metric_risk(predictions, targets):
            string_targets = list(tokenizer.ids2string(t) for t in targets)
            string_predictions = list(tokenizer.ids2string(t) for t in predictions)
            metrics = [external_metrics_func([t], [p], epoch=-1, metric=risk_metric) for p, t in
                       zip(string_predictions, string_targets)]

            if any([s in risk_metric for s in ['entropy', 'nist', 'avg_len']]):
                return [-m for m in metrics]

            assert all([0 <= s <= 1.0 for s in metrics]), metrics

            return [1 - m for m in metrics]

        if risk_metric == 'f1':
            return f1_risk

        return external_metric_risk

    # helpers -----------------------------------------------------

    try:
        model_trainer.train(after_epoch_funcs=[save_func, sample_text_func, test_func],
                            risk_func=get_risk_metric_func(trainer_config.risk_metric))
    except (KeyboardInterrupt, Exception, RuntimeError) as e:
        raise e

def main():
    args = InputConfig().args

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    trainer_config = get_trainer_config(args)

    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    comment = '_{}_{}_single'.format(args.model_type, args.data_type)
    logdir = os.path.join('runs', current_time + comment)
    writer = SummaryWriter(logdir=logdir)
    logger = config_logger(os.path.join(logdir, 'train.log'))

    log_dir = writer.logdir
    logger.info("Training args: {}".format(args))
    interrupt_checkpoint_path = os.path.join(log_dir, trainer_config.interrupt_checkpoint_path)
    last_checkpoint_path = os.path.join(log_dir, trainer_config.last_checkpoint_path)
    best_checkpoint_path = os.path.join(log_dir, 'best_model')
    logger.info("Logging to {}".format(log_dir))  # Let's save everything on an experiment in the ./runs/XXX/directory

    set_seed(trainer_config.seed)
    device = torch.device(trainer_config.device)

    model, tokenizer = get_model_and_tokenizer(args, trainer_config, logger)
    logger.info('Load tokenizer, vocab size is %d', tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else
            tokenizer.n_words)
    modify_model(args, model, tokenizer)
    training_procedure(args, trainer_config, model, tokenizer, device, writer, logger,
                       best_checkpoint_path,
                       last_checkpoint_path, interrupt_checkpoint_path, log_dir, test_data_type=args.test_data_type)

if __name__ == '__main__':
    main()
