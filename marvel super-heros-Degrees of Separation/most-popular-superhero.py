from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster("local").setAppName("PopularHero")
sc = SparkContext(conf = conf)

def countCoOccurences(line):
    elements = line.split()
    return (int(elements[0]), len(elements) - 1)

def parseNames(line):
    fields = line.split('\"')
    return (int(fields[0]), fields[1].encode("utf8"))

def flipRdd(x) : 
	return (x[1],x[0])

names = sc.textFile(r"C:\Users\sraghunath\Desktop\spark_course\Marvel-Names.txt")
namesRdd = names.map(parseNames)

lines = sc.textFile(r"C:\Users\sraghunath\Desktop\spark_course\Marvel-Graph.txt")

pairings = lines.map(countCoOccurences)
totalFriendsByCharacter = pairings.reduceByKey(lambda x, y : x + y)
flipped = totalFriendsByCharacter.map(flipRdd)

mostPopular = flipped.max()

mostPopularName = namesRdd.lookup(mostPopular[1])[0]

print(mostPopular)