package sentiment

import scalanlp.text.tokenize._

import scala.collection.JavaConversions._
import scala.io.Source

import java.io.File


object Sentiment {

  object SentimentLabel extends Enumeration {
    type SentimentLabel = Value
    val Pos, Neg = Value
  }
  import SentimentLabel

  val dataPath = "./review_polarity/txt_sentoken"

  val tokenizer = SimpleEnglishTokenizer.apply()

  def tokenizeFiles(files: Seq[File]) = {
    for (f <- files) yield tokenizer(Source.fromFile(f).mkString).toIndexedSeq
  }

  def main(args: Array[String]) {
    val neg = tokenizeFiles(new File(dataPath, "neg").listFiles)
    val pos = tokenizeFiles(new File(dataPath, "pos").listFiles)

  }

}

