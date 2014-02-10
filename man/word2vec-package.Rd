\name{word2vec-package}
\alias{word2vec-package}
\alias{word2vec}
\docType{package}
\title{
R interface to Google's word2vec library
word2vec
}
\description{
This package exposes Google's word2vec library to R.
word2vec provides an efficient implementation of the continuous bag-of-words and skip-gram architectures for computing vector representations of words. These representations can be subsequently used in many natural language processing applications and for further research. 
}
\details{
\tabular{ll}{
Package: \tab word2vec\cr
Type: \tab Package\cr
Version: \tab 1.0\cr
Date: \tab 2014-02-07\cr
License: \tab Apache License Version 2.0\cr
}
}
\author{
Thanh Dinh Khac

Maintainer: Thanh Dinh Khac <thanh.dinhkhac@revolutionanalytics.com>
}
\references{
~~ Literature or other references for background information ~~
}
~~ Optionally other standard keywords, one per line, from file KEYWORDS in ~~
~~ the R documentation directory ~~
\keyword{ package }
\seealso{
~~ \code{\link[<pkg>:<pkg>-package]{<pkg>}} ~~
}
\examples{
	v <- read.vectors('vectors.bin')
}
