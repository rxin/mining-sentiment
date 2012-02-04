package sentiment

import scalanlp.text.tokenize._

import scala.collection.JavaConversions._
import scala.io.Source

import java.io.File

import scalala.tensor.mutable._

import scala.math.log
import scala.util.Random
import scalanlp.data._

object Sentiment {

  object SentimentLabel extends Enumeration {
    type SentimentLabel = Value
    val Pos, Neg = Value
  }
  import SentimentLabel._

  // TODO: index words

  type Word = String
  type Document = Seq[String]

  val masterRandom = Random

  case class Model(counts: Counter2[SentimentLabel, Word, Double],
                   labelCounts : Counter[SentimentLabel, Double],
                   alpha: Double) {
    val wordSet = new scala.collection.mutable.HashSet[Word]

    def score(label: SentimentLabel, document: Document): Double = {
      var score: Double = 0
      score += log(labelCounts(label) + alpha)
      score -= document.length * log((labelCounts(label) + wordSet.size * alpha))
      for (word <- document) {
        score += log(counts(label, word) + alpha)
      }
      return score
    }

    def classify(document: Document): SentimentLabel = {
      val negScore = score(Neg, document)
      val posScore = score(Pos, document)
      return if (negScore < posScore) Pos else Neg
    }

    def trainOne(label: SentimentLabel, words: Document) {
      wordSet ++= words
      for (word <- words) {
        counts(label, word) += 1
        labelCounts(label) += 1
      }
    }
  }

  val dataPath = "./review_polarity/txt_sentoken"

  val tokenizer = SimpleEnglishTokenizer.apply()

  def tokenizeFiles(files: Seq[File]) = {
    for (f <- files) yield {
      val tokens = tokenizer(Source.fromFile(f).mkString).toIndexedSeq
      Pair(f getName, tokens)
    }
  }

  def train(trainingData: Seq[LabeledDocument[SentimentLabel, Word]]) = {
    val model = Model(Counter2(), Counter(), 1)
    for (train <- trainingData) model.trainOne(train.label, train.features("body"))
    model
  }

  def f1Score(model: Model, testData: Seq[LabeledDocument[SentimentLabel, Word]],
              label: SentimentLabel): Double = {
    val counts = testData.map { doc =>
      val docLabel = model.classify(doc.features("body"))
      if (docLabel == doc.label) {
        if (doc.label == label) {
          "tp"
        } else {
          "tn"
        }
      } else {
        if (doc.label == label) {
          "fp"
        } else {
          "fn"
        }
      }
    }.groupBy(x => x) map { case (key, value) => (key, value size) }
    val betaSq: Double = 1
    return (1 + betaSq) * counts("tp") /
           ((1 + betaSq) * counts("tp") + betaSq * counts("fn") + counts("fp"))
  }

  // Instructor recommends evaluating using F1 or AUC.
  def evaluate(model: Model, testData: Seq[LabeledDocument[SentimentLabel, Word]]) = {
    (f1Score(model, testData, Pos) + f1Score(model, testData, Neg)) * 0.5
  }

  def getLabeledDocuments(files : Seq[File], label : SentimentLabel) = {
    tokenizeFiles(files).map {
      case Pair(name, words) => new LabeledDocument(name, label, Map("body" -> words))
    }
  }

  def main(args: Array[String]) {
    val neg = getLabeledDocuments(new File(dataPath, "neg").listFiles, Neg)
    val pos = getLabeledDocuments(new File(dataPath, "pos").listFiles, Pos)

    val all = masterRandom.shuffle(neg ++ pos)
    val sliceSize = all.size / 10
    val scores = (0 until 10).par map { i => {
      val a = i * sliceSize
      val b = (i + 1) * sliceSize
      val testData = all.slice(a, b)
      val trainingData = all.take(a) ++ all.drop(b)
      val model = train(trainingData)
      val score = evaluate(model, testData)
      println(score)
      score
    }}
    println("Congratulations!  Your average score is " + scores.sum / scores.size)
  }

}

