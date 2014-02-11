\title{Train a vector representation for a word corpus}
\name{train.vectors}
\alias{train.vectors}
\description{
	This function generate vector representation from a text input file.
}
\usage{
train.vectors(train.file, output.file="vectors.bin", size=100, cbow=0, hs=1, negative=0, min.count=5, alpha=0.025)
}
\arguments{
	\item{train.file}{Name of the training text file}
	\item{output.file}{Name of the output file that contains trained vectors}
	\item{size}{Size of the trained vectors}
	\item{cbow}{Use the continuous bag of word model, default is 0 (skip-gram model)}
	\item{hs}{Use hieriarchial Softmax, default is 1, 0 means not used}
	\item{negative}{Number of negative examples, default is 0 (not used), common values are 5-10}
	\item{min.count}{Discard words that appear less than min.count times, default is 5}
	\item{alpha}{Starting learning rate, default is 0.025}
}
\details{
}
\value{
	Return 0 if successful.
}
\examples{
	train.vectors(train.file="text8");

	train.vectors(train.file="text8", output.file="vectors.bin", size=150, cbow=0, hs=1,
			  negative=0, min.count=5, alpha=0.025);
}
