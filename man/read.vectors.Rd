\title{Read a vector representation from file}
\name{read.vectors}
\alias{read.vectors}
\description{
	This function read vector representation generated from \code{train.vectors}
}
\usage{
read.vectors(vector.file)
}
\arguments{
	\item{vector.file}{Name of the file contains word vectors}
}
\details{
}
\value{
	Return list of vectors read from input file with names correspond to the words
}
\examples{
	v <- read.vectors("vectors.bin");
}
