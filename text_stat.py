import sys
import os.path
from pyspark import SparkContext
from pyspark.sql import Row
from pyspark.sql import SparkSession

sc = SparkContext()


def remove_punctuation(partition):
    punctuation = [',', '.', ';', ':', '?', '!', '"', '-', '\'']
    for word in partition:
        if any(x == word[0] for x in punctuation):
            yield word[1:]
        elif any(x == word[-1] for x in punctuation):
            yield word[:-1]
        else:
            yield word


def load_dataframe(file_name, spark, cores):
    if cores.isdigit():
        words = sc.textFile(file_name, minPartitions=int(cores)) \
            .flatMap(lambda line: line.split()) \
            .mapPartitions(remove_punctuation) \
            .filter(lambda word: word is not None and len(word) > 0) \
            .filter(lambda word: word.count("@") == 0) \
            .filter(lambda word: word.count("//") == 0) \
            .filter(lambda word: word.count("---") == 0) \
            .map(lambda word: word.lower()) \
            .persist()
    else:
        words = sc.textFile(file_name) \
            .flatMap(lambda line: line.split()) \
            .mapPartitions(remove_punctuation) \
            .filter(lambda word: word is not None and len(word) > 0) \
            .filter(lambda word: word.count("@") == 0) \
            .filter(lambda word: word.count("//") == 0) \
            .filter(lambda word: word.count("---") == 0) \
            .map(lambda word: word.lower()) \
            .persist()

    word_count = words.count()

    word_freq = words.map(lambda word: (word, 1)) \
        .reduceByKey(lambda count1, count2: count1 + count2) \
        .map(lambda word_n_count: (word_n_count[0], word_n_count[1] / float(word_count)))

    rdd = word_freq.map(lambda line: Row(text_word=line[0], word_freq=line[1], word_length=len(line[0])))

    if cores.isdigit():
        return spark.createDataFrame(rdd).coalesce(int(cores))

    return spark.createDataFrame(rdd)


def usage(prog):
    print("Usage: " + prog + " --file <text file to analyze> [--cores <number of cores>]")


def main():
    if len(sys.argv) != 3 and len(sys.argv) != 5:
        usage(sys.argv[0])
        sys.exit(-1)

    file_name = ''
    cores = '*'

    flags = ['--file', '--cores']
    if (len(sys.argv) == 3) and (sys.argv[1] not in flags):
        usage(sys.argv[0])
        sys.exit(-1)
    elif (len(sys.argv) == 5) and ((sys.argv[1] not in flags) or (sys.argv[3] not in flags)):
        usage(sys.argv[0])
        sys.exit(-1)

    i = 0
    while i < len(sys.argv):
        if sys.argv[i] == '--file':
            i += 1
            file_name = sys.argv[i]
            if not os.path.isfile(file_name):
                print("Unable to find file: " + file_name)
                sys.exit(-1)
        if sys.argv[i] == '--cores':
            i += 1
            cores = sys.argv[i]
            if not cores.isdigit():
                print("number of cores '{}' must be an integer".format(cores))
                sys.exit(-1)
        i += 1

    if len(file_name) == 0:
        usage(sys.argv[0])
        sys.exit(-1)

    master = "local[" + cores + "]"
    if cores.isdigit():
        spark = SparkSession \
            .builder \
            .master(master) \
            .appName("Text Statistics") \
            .config("spark.executor.memory", "1g") \
            .config("spark.cores.max", cores) \
            .getOrCreate()
    else:
        spark = SparkSession \
            .builder \
            .master(master) \
            .appName("Text Statistics") \
            .config("spark.executor.memory", "1g") \
            .getOrCreate()

    df = load_dataframe(file_name, spark, cores)

    longest_word = df.select("text_word", "word_length").sort(df.word_length.desc()).limit(1).collect()[0][0]
    freq_4_letter = df.select("text_word", "word_freq", "word_length").filter(df.word_length == 4) \
        .sort(df.word_freq.desc()).limit(1).collect()[0][0]
    freq_15_letter = df.select("text_word", "word_freq", "word_length").filter(df.word_length == 15) \
        .sort(df.word_freq.desc()).limit(1).collect()[0][0]

    print("Longest word: {}".format(longest_word))
    print("Most frequent 4-letter word: {}".format(freq_4_letter))
    print("Most frequent 15-letter word: {}".format(freq_15_letter))


if __name__ == "__main__":
    main()
    # input("ctrl c to stop")
