import logging
from pathlib import Path

import click
from flair.data import Sentence
from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger


# Convert IOBES to IOB for CoNLL evaluations script
def iobes_to_iob(tag):
    iob_tag = tag

    if tag.startswith("S-"):
        iob_tag = tag.replace("S-", "B-")

    if tag.startswith("E-"):
        iob_tag = tag.replace("E-", "I-")

    return iob_tag


@click.command()
@click.option("--dataset", type=str, help="Define dataset")
@click.option("--model", type=click.Path(exists=True))
def parse_arguments(dataset, model):
    # Adjust logging level
    logging.getLogger("flair").setLevel(level="ERROR")

    columns = {0: "text", 1: "ner"}

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

    tagger: SequenceTagger = SequenceTagger.load(model)

    for test_sentence in corpus.test:
        tokens = test_sentence.tokens
        gold_tags = [token.tags["ner"].value for token in tokens]

        tagged_sentence = Sentence()
        tagged_sentence.tokens = tokens

        # Tag sentence with model
        tagger.predict(tagged_sentence)

        predicted_tags = [token.tags["ner"].value for token in tagged_sentence.tokens]

        assert len(tokens) == len(gold_tags)
        assert len(gold_tags) == len(predicted_tags)

        for token, gold_tag, predicted_tag in zip(tokens, gold_tags, predicted_tags):
            gold_tag = iobes_to_iob(gold_tag)
            predicted_tag = iobes_to_iob(predicted_tag)

            print(f"{token.text} {gold_tag} {predicted_tag}")

        print("")


if __name__ == "__main__":
    parse_arguments()
