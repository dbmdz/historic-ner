import json
from pathlib import Path
from typing import List

import click
from flair.datasets import ColumnCorpus
from flair.embeddings import (
    TokenEmbeddings,
    WordEmbeddings,
    StackedEmbeddings,
    CharacterEmbeddings,
    BertEmbeddings,
    FlairEmbeddings,
    BytePairEmbeddings,
)
from flair import logger
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from gensim.models import FastText


def run_experiment(number, corpus, embedding_types, run, experiment_details):
    tag_type = "ner"
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    logger.info(corpus)

    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

    tagger: SequenceTagger = SequenceTagger(
        hidden_size=512,
        embeddings=embeddings,
        tag_dictionary=tag_dictionary,
        tag_type=tag_type,
        use_crf=True,
    )

    trainer: ModelTrainer = ModelTrainer(tagger, corpus)

    trainer.train(
        f"resources/taggers/experiment_{number}_{run}",
        learning_rate=0.1,
        mini_batch_size=8,
        max_epochs=500,
    )

    with open(f"resources/taggers/experiment_{number}_{run}/details.json", "w") as f_p:
        json.dump(experiment_details, f_p)


@click.command()
@click.option("--number", type=int, help="Define experiment number")
@click.option("--dataset", type=str, help="Define dataset")
@click.option("--embeddings", type=str, help="Comma separated list of embedding types")
@click.option("--lms", type=str, help="Comma separated list of language models")
@click.option("--runs", type=int, default=3, help="Define number of runs")
def parse_arguments(number, dataset, embeddings, lms, runs):
    number = number

    columns = {0: "text", 1: "ner"}
    embedding_types: List[TokenEmbeddings] = []

    if dataset == "lft":
        corpus: ColumnCorpus = ColumnCorpus(
            Path("./data"),
            columns,
            train_file="./enp_DE.lft.mr.tok.train.bio",
            dev_file="./enp_DE.lft.mr.tok.dev.bio",
            test_file="./enp_DE.lft.mr.tok.test.bio",
            tag_to_bioes="ner",
        )
    elif dataset == "onb":
        corpus: ColumnCorpus = ColumnCorpus(
            Path("./data"),
            columns,
            train_file="./enp_DE.onb.mr.tok.train.bio",
            dev_file="./enp_DE.onb.mr.tok.dev.bio",
            test_file="./enp_DE.onb.mr.tok.test.bio",
            tag_to_bioes="ner",
        )

    for embedding in embeddings.split(","):
        if embedding == "wikipedia":
            logger.info("Using Wikipedia FastText embeddings")
            embedding_types.append(WordEmbeddings("de"))
        elif embedding == "wikipedia_subword":
            model = FastText.load_fasttext_format("wiki.de")
            word_vectors = model.wv

            # Save them into a flair-importable format
            word_vectors.save("wiki.de.vec.gensim")

            logger.info("Using Wikipedia FastText embeddings with subword information")
            embedding_types.append(WordEmbeddings("wiki.de.vec.gensim"))
        elif embedding == "character":
            logger.info("Using character embeddings")
            embedding_types.append(CharacterEmbeddings())
        elif embedding == "crawl":
            logger.info("Using CommonCrawl FastText embeddings")
            embedding_types.append(WordEmbeddings("de-crawl"))
        elif embedding == "crawl_subword":
            model = FastText.load_fasttext_format("cc.de.300")
            word_vectors = model.wv

            # Save them into a flair-importable format
            word_vectors.save("cc.de.300.vec.gensim")

            logger.info(
                "Using CommonCrawl FastText embeddings with subword information"
            )
            embedding_types.append(WordEmbeddings("cc.de.300.vec.gensim"))
        elif embedding == "bpe":
            logger.info("Using German BPEmbeddings with 100k vocab and 300 dims")
            embedding_types.append(BytePairEmbeddings(language="de", dim=300))
        else:
            logger.error(
                f"Embedding name {embedding} not recognized! Please check your input!"
            )
            exit(1)

    for lm in lms.split(","):
        if lm == "de":
            logger.info("Using German language model")
            embedding_types.append(FlairEmbeddings("de-forward"))
            embedding_types.append(FlairEmbeddings("de-backward"))
        elif lm == "hamburger_anzeiger":
            logger.info("Using Hamburger Anzeiger language model")
            embedding_types.append(FlairEmbeddings("de-historic-ha-forward"))
            embedding_types.append(FlairEmbeddings("de-historic-ha-backward"))
        elif lm == "wiener_zeitung":
            logger.info("Using Wiener Zeitung language model")
            embedding_types.append(FlairEmbeddings("de-historic-wz-forward"))
            embedding_types.append(FlairEmbeddings("de-historic-wz-backward"))
        elif lm == "hamburger_anzeiger_pooled":
            logger.info("Using Hamburger Anzeiger language model (pooled)")
            embedding_types.append(
                PooledFlairEmbeddings("de-historic-ha-forward", pooling="min")
            )
            embedding_types.append(
                PooledFlairEmbeddings("de-historic-ha-backward", pooling="min")
            )
        elif lm == "wiener_zeitung_pooled":
            logger.info("Using Wiener Zeitung language model (pooled)")
            embedding_types.append(
                PooledFlairEmbeddings("de-historic-wz-forward", pooling="min")
            )
            embedding_types.append(
                PooledFlairEmbeddings("de-historic-wz-backward", pooling="min")
            )
        elif lm.startswith("bert"):
            # bert-base-multilingual-cased_-4;-3;-2;-1 -> parse layers
            layers = lm.split("_")[1].replace(";", ",")
            bert_model = lm.split("_")[0]

            logger.info("Using BERT multi-language model with layers:", layers)
            embedding_types.append(
                BertEmbeddings(
                    bert_model_or_path="bert-base-multilingual-cased", layers=layers
                )
            )
        else:
            logger.info(f"Language model {lm} not recognized! Could be ok...")

    experiment_details = {
        "number": number,
        "dataset": dataset,
        "embeddings": embeddings,
        "lms": lms,
    }

    for run in range(runs):
        run_experiment(number, corpus, embedding_types, run, experiment_details)


if __name__ == "__main__":
    parse_arguments()
