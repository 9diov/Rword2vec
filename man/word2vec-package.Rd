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
~~ An overview of how to use the package, including the most important ~~
~~ functions ~~
}
\author{
Thanh Dinh Khac

Maintainer: Who to complain to <yourfault@somewhere.net>
}
\references{
~~ Literature or other references for background information ~~
}
~~ Optionally other standard keywords, one per line, from file KEYWORDS in ~~
~~ the R documentation directory ~~
\keyword{ package }
\seealso{
~~ Optional links to other man pages, e.g. ~~
~~ \code{\link[<pkg>:<pkg>-package]{<pkg>}} ~~
}
\examples{
	v <- read.vectors('vectors.bin')
}
