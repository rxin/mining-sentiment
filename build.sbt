
name := "sentiment"

version := "1.0"

organization := "org.sentiment"

scalaVersion := "2.9.1"

libraryDependencies  ++= Seq(
  "org.scalala" %% "scalala" % "1.0.0.RC3-SNAPSHOT",
  "org.scalanlp" %% "scalanlp-learn" % "0.5-SNAPSHOT"
)


resolvers ++= Seq(
  "Typesafe" at "http://repo.typesafe.com/typesafe/releases",
  "Scala Tools Snapshots" at "http://scala-tools.org/repo-snapshots/",
  "ScalaNLP Maven2" at "http://repo.scalanlp.org/repo"
)

