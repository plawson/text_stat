import sys
import os.path
from pyspark import SparkContext
from pyspark.sql import Row
from pyspark.sql import SparkSession

sc = SparkContext()
spark = SparkSession \
    .builder \
    .master("local[4") \
    .appName("Text Statistics") \
    .config("spark.executor.memory", "2g") \
    .config("spark.cores.max", "4") \
    .getOrCreate()


def remove_punctuation(partition):
    for word in partition:
        if word[0] == ',' \
                or word[0] == '.' \
                or word[0] == ';' \
                or word[0] == ':' \
                or word[0] == '?' \
                or word[0] == '!' \
                or word[0] == '"' \
                or word[0] == '-' \
                or word[0] == '\'':
            yield word[1:]

        elif word[len(word) - 1] == '.' \
                or word[len(word) - 1] == ',' \
                or word[len(word) - 1] == ';' \
                or word[len(word) - 1] == ':' \
                or word[len(word) - 1] == '?' \
                or word[len(word) - 1] == '!' \
                or word[len(word) - 1] == '"' \
                or word[len(word) - 1] == '-' \
                or word[len(word) - 1] == '\'':
            yield word[:-1]

        else:
            yield word


def load_dataframe(file_name):
    words = sc.textFile(file_name, minPartitions=4) \
        .flatMap(lambda line: line.split()) \
        .mapPartitions(remove_punctuation) \
        .filter(lambda word: word is not None and len(word) > 0) \
        .filter(lambda word: word.count("@") == 0) \
        .filter(lambda word: word.count("//") == 0) \
        .filter(lambda word: word.count("---") == 0) \
        .map(lambda word: word.lower())

    word_count = words.count()

    word_freq = words.map(lambda word: (word, 1)) \
        .reduceByKey(lambda count1, count2: count1 + count2) \
        .map(lambda word_n_count: (word_n_count[0], word_n_count[1] / float(word_count)))

    rdd = word_freq.map(lambda line: Row(text_word=line[0], word_freq=line[1], word_length=-len(line[0])))

    return spark.createDataFrame(rdd)


def main():
    if len(sys.argv) != 2:
        print("Usage: " + sys.argv[0] + " <text file to analyze>")
        sys.exit(-1)

    file_name = sys.argv[1]
    if not os.path.isfile(file_name):
        print("Unable to find file: " + file_name)
        sys.exit(-1)

    words = load_dataframe(file_name)

    words.select("text_word", "word_freq", "word_length").sort("word_length").show(truncate=False)


if __name__ == "__main__":
    main()
