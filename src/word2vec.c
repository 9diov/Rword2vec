//  Copyright 2013 Google Inc. All Rights Reserved.
//  Copyright 2014 Revolution Analytics Pte Ltd.
//  Modified to be more suitable to be part of a R package
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <R.h>
#include <Rinternals.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

const int VOCAB_HASH_SIZE = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

const long long MAX_SIZE = 2000;       // max length of strings
const long long N = 40;                // number of closest words that will be shown
const long long MAX_W = 50;            // max length of vocabulary entries

const int TABLE_SIZE = 1e8;            // size of unigram table

struct internal_data {
	char *vocab;
	float *M;
	long long words, size; // words: vocab size, size: vector dimensions
};

struct configs {
	long long layer1_size, classes, file_size, train_words, word_count_actual;
	char train_file[MAX_STRING], output_file[MAX_STRING];
	char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
	int debug_mode, binary, cbow, window, min_count, num_threads, hs, negative;
	real alpha, starting_alpha, sample;
	clock_t start;
};

static void InitConfigs(struct configs *o) {
	o->num_threads = 1;
	o->classes = 0;
	o->file_size = 0;
	o->train_words = 0;
	o->word_count_actual = 0;
	o->layer1_size = 100;
	o->debug_mode = 2;
	o->binary = 1;
	o->cbow = 0;
	o->hs = 1;
	o->alpha =0.025;
	o->sample = 0;
	o->output_file[0] = 0;
	o->save_vocab_file[0] = 0;
	o->read_vocab_file[0] = 0;
	o->window = 5;
	o->min_count = 5;
}

struct vocab_word {
	long long cn;
	int *point;
	char *word, *code, codelen;
};

struct vocab {
	struct vocab_word *vocab;
	long long vocab_size;
	long long vocab_MAX_SIZE;
	int min_reduce;
	int *vocab_hash;
};

static void InitVocab(struct vocab* v) {
	v->vocab_MAX_SIZE = 1000;
	v->min_reduce = 1;
	v->vocab = (struct vocab_word *)calloc(v->vocab_MAX_SIZE, sizeof(struct vocab_word));
	v->vocab_hash = (int *)calloc(VOCAB_HASH_SIZE, sizeof(int));
}

struct net {
	real *syn0, *syn1, *syn1neg, *expTable;
};

static void InitNetData(struct net *n) {
	int i;
	n->expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
	for (i = 0; i < EXP_TABLE_SIZE; i++) {
		n->expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
		n->expTable[i] = n->expTable[i] / (n->expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
	}
}

struct all_data {
	struct vocab *v;
	struct net *n;
	int *table;
	struct configs *o;
	long id;
};

static void InitUnigramTable(int *table, struct vocab *v) {
	int a, i;
	long long train_words_pow = 0;
	real d1, power = 0.75;
	table = (int *)malloc(TABLE_SIZE * sizeof(int));
	for (a = 0; a < v->vocab_size; a++) train_words_pow += pow(v->vocab[a].cn, power);
	i = 0;
	d1 = pow(v->vocab[i].cn, power) / (real)train_words_pow;
	for (a = 0; a < TABLE_SIZE; a++) {
		table[a] = i;
		if (a / (real)TABLE_SIZE > d1) {
			i++;
			d1 += pow(v->vocab[i].cn, power) / (real)train_words_pow;
		}
		if (i >= v->vocab_size) i = v->vocab_size - 1;
	}
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
static void ReadWord(char *word, FILE *fin) {
	int a = 0, ch;
	while (!feof(fin)) {
		ch = fgetc(fin);
		if (ch == 13) continue;
		if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
			if (a > 0) {
				if (ch == '\n') ungetc(ch, fin);
				break;
			}
			if (ch == '\n') {
				strcpy(word, (char *)"</s>");
				return;
			} else continue;
		}
		word[a] = ch;
		a++;
		if (a >= MAX_STRING - 1) a--;   // Truncate too long words
	}
	word[a] = 0;
}

// Returns hash value of a word
int GetWordHash(char *word) {
	unsigned long long a, hash = 0;
	for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
	hash = hash % VOCAB_HASH_SIZE;
	return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word, struct vocab *v) {
	unsigned int hash = GetWordHash(word);
	while (1) {
		if (v->vocab_hash[hash] == -1) return -1;
		if (!strcmp(word, v->vocab[v->vocab_hash[hash]].word)) return v->vocab_hash[hash];
		hash = (hash + 1) % VOCAB_HASH_SIZE;
	}
	return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin, struct vocab *v) {
	char word[MAX_STRING];
	ReadWord(word, fin);
	if (feof(fin)) return -1;
	return SearchVocab(word, v);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word, struct vocab *v) {
	unsigned int hash, length = strlen(word) + 1;
	if (length > MAX_STRING) length = MAX_STRING;
	v->vocab[v->vocab_size].word = (char *)calloc(length, sizeof(char));
	strcpy(v->vocab[v->vocab_size].word, word);
	v->vocab[v->vocab_size].cn = 0;
	v->vocab_size++;
	// Reallocate memory if needed
	if (v->vocab_size + 2 >= v->vocab_MAX_SIZE) {
		v->vocab_MAX_SIZE += 1000;
		v->vocab = (struct vocab_word *)realloc(v->vocab, v->vocab_MAX_SIZE * sizeof(struct vocab_word));
	}
	hash = GetWordHash(word);
	while (v->vocab_hash[hash] != -1) hash = (hash + 1) % VOCAB_HASH_SIZE;
	v->vocab_hash[hash] = v->vocab_size - 1;
	return v->vocab_size - 1;
}

// Used later for sorting by word counts
static int VocabCompare(const void *a, const void *b) {
	return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
static void SortVocab(struct vocab *v, struct configs *o) {
	int a, size;
	unsigned int hash;
	// Sort the vocabulary and keep </s> at the first position
	qsort(&(v->vocab[1]), v->vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
	for (a = 0; a < VOCAB_HASH_SIZE; a++) v->vocab_hash[a] = -1;
	size = v->vocab_size;
	o->train_words = 0;
	for (a = 0; a < size; a++) {
		// Words occuring less than min_count times will be discarded from the vocab
		if (v->vocab[a].cn < o->min_count) {
			v->vocab_size--;
			free(v->vocab[v->vocab_size].word);
		} else {
			// Hash will be re-computed, as after the sorting it is not actual
			hash=GetWordHash(v->vocab[a].word);
			while (v->vocab_hash[hash] != -1) hash = (hash + 1) % VOCAB_HASH_SIZE;
			v->vocab_hash[hash] = a;
			o->train_words += v->vocab[a].cn;
		}
	}
	v->vocab = (struct vocab_word *)realloc(v->vocab, (v->vocab_size + 1) * sizeof(struct vocab_word));
	// Allocate memory for the binary tree construction
	for (a = 0; a < v->vocab_size; a++) {
		v->vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
		v->vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
	}
}

// Reduces the vocabulary by removing infrequent tokens
static void ReduceVocab(struct vocab *v) {
	int a, b = 0;
	unsigned int hash;
	for (a = 0; a < v->vocab_size; a++) if (v->vocab[a].cn > v->min_reduce) {
		v->vocab[b].cn = v->vocab[a].cn;
		v->vocab[b].word = v->vocab[a].word;
		b++;
	} else free(v->vocab[a].word);
	v->vocab_size = b;
	for (a = 0; a < VOCAB_HASH_SIZE; a++) v->vocab_hash[a] = -1;
	for (a = 0; a < v->vocab_size; a++) {
		// Hash will be re-computed, as it is not actual
		hash = GetWordHash(v->vocab[a].word);
		while (v->vocab_hash[hash] != -1) hash = (hash + 1) % VOCAB_HASH_SIZE;
		v->vocab_hash[hash] = a;
	}
	//fflush(stdout);
	v->min_reduce++;
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
static void CreateBinaryTree(struct vocab *v) {
	long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
	char code[MAX_CODE_LENGTH];
	long long *count = (long long *)calloc(v->vocab_size * 2 + 1, sizeof(long long));
	long long *binary = (long long *)calloc(v->vocab_size * 2 + 1, sizeof(long long));
	long long *parent_node = (long long *)calloc(v->vocab_size * 2 + 1, sizeof(long long));
	for (a = 0; a < v->vocab_size; a++) count[a] = v->vocab[a].cn;
	for (a = v->vocab_size; a < v->vocab_size * 2; a++) count[a] = 1e15;
	pos1 = v->vocab_size - 1;
	pos2 = v->vocab_size;
	// Following algorithm constructs the Huffman tree by adding one node at a time
	for (a = 0; a < v->vocab_size - 1; a++) {
		// First, find two smallest nodes 'min1, min2'
		if (pos1 >= 0) {
			if (count[pos1] < count[pos2]) {
				min1i = pos1;
				pos1--;
			} else {
				min1i = pos2;
				pos2++;
			}
		} else {
			min1i = pos2;
			pos2++;
		}
		if (pos1 >= 0) {
			if (count[pos1] < count[pos2]) {
				min2i = pos1;
				pos1--;
			} else {
				min2i = pos2;
				pos2++;
			}
		} else {
			min2i = pos2;
			pos2++;
		}
		count[v->vocab_size + a] = count[min1i] + count[min2i];
		parent_node[min1i] = v->vocab_size + a;
		parent_node[min2i] = v->vocab_size + a;
		binary[min2i] = 1;
	}
	// Now assign binary code to each vocabulary word
	for (a = 0; a < v->vocab_size; a++) {
		b = a;
		i = 0;
		while (1) {
			code[i] = binary[b];
			point[i] = b;
			i++;
			b = parent_node[b];
			if (b == v->vocab_size * 2 - 2) break;
		}
		v->vocab[a].codelen = i;
		v->vocab[a].point[0] = v->vocab_size - 2;
		for (b = 0; b < i; b++) {
			v->vocab[a].code[i - b - 1] = code[b];
			v->vocab[a].point[i - b] = point[b] - v->vocab_size;
		}
	}
	free(count);
	free(binary);
	free(parent_node);
}

int LearnVocabFromTrainFile(struct vocab *v, struct configs *o) {
	char word[MAX_STRING];
	FILE *fin;
	long long a, i;
	for (a = 0; a < VOCAB_HASH_SIZE; a++) v->vocab_hash[a] = -1;
	fin = fopen(o->train_file, "rb");
	if (fin == NULL) {
		Rprintf("ERROR: training data file not found!\n");
		return 1;
	}
	v->vocab_size = 0;
	AddWordToVocab((char *)"</s>", v);
	while (1) {
		ReadWord(word, fin);
		if (feof(fin)) break;
		(o->train_words)++;
		if ((o->debug_mode > 1) && (o->train_words % 100000 == 0)) {
			Rprintf("%lldK%c", o->train_words / 1000, 13);
			//fflush(stdout);
		}
		i = SearchVocab(word, v);
		if (i == -1) {
			a = AddWordToVocab(word, v);
			v->vocab[a].cn = 1;
		} else v->vocab[i].cn++;
		if (v->vocab_size > VOCAB_HASH_SIZE * 0.7) ReduceVocab(v);
	}
	if (o->debug_mode > 0) {
		Rprintf("Vocab size: %lld\n", v->vocab_size);
		Rprintf("Words in train file: %lld\n", o->train_words);
	}
	SortVocab(v, o);
	if (o->debug_mode > 0) {
		Rprintf("Vocab size: %lld\n", v->vocab_size);
		Rprintf("Words in train file: %lld\n", o->train_words);
	}
	o->file_size = ftell(fin);
	fclose(fin);
	Rprintf("Learn vocab done\n");
	return 0;
}

static void SaveVocab(struct vocab *v, struct configs *o) {
	long long i;
	FILE *fo = fopen(o->save_vocab_file, "wb");
	for (i = 0; i < v->vocab_size; i++) fprintf(fo, "%s %lld\n", v->vocab[i].word, v->vocab[i].cn);
	fclose(fo);
}

static void ReadVocab(struct vocab *v, struct configs *o) {
	long long a, i = 0;
	char c;
	char word[MAX_STRING];
	FILE *fin = fopen(o->read_vocab_file, "rb");
	if (fin == NULL) {
		Rprintf("Vocabulary file not found\n");
		exit(1);
	}
	for (a = 0; a < VOCAB_HASH_SIZE; a++) v->vocab_hash[a] = -1;
	v->vocab_size = 0;
	while (1) {
		ReadWord(word, fin);
		if (feof(fin)) break;
		a = AddWordToVocab(word, v);
		fscanf(fin, "%lld%c", &(v->vocab[a].cn), &c);
		i++;
	}
	if (o->debug_mode > 0) {
		Rprintf("Vocab size: %lld\n", v->vocab_size);
		Rprintf("Words in train file: %lld\n", o->train_words);
	}
	SortVocab(v, o);
	if (o->debug_mode > 0) {
		Rprintf("Vocab size: %lld\n", v->vocab_size);
		Rprintf("Words in train file: %lld\n", o->train_words);
	}
	fin = fopen(o->train_file, "rb");
	if (fin == NULL) {
		Rprintf("ERROR: training data file not found!\n");
		exit(1);
	}
	fseek(fin, 0, SEEK_END);
	o->file_size = ftell(fin);
	fclose(fin);
}

static void InitNet(struct net *n, struct vocab *v, struct configs *o) {
	long long a, b;
#ifdef _WIN32
	n->syn0 = (real *)malloc((long long)v->vocab_size * o->layer1_size * sizeof(real));
#elif __posix
	a = posix_memalign((void **)&(n->syn0), 128, (long long)v->vocab_size * o->layer1_size * sizeof(real));
#endif
	if (n->syn0 == NULL) {Rprintf("Memory allocation failed\n"); exit(1);}
	if (o->hs) {
#ifdef _WIN32
		n->syn1 = (real *)malloc((long long)v->vocab_size * o->layer1_size * sizeof(real));
#elif __posix
		a = posix_memalign((void **)&(n->syn1), 128, (long long)v->vocab_size * o->layer1_size * sizeof(real));
#endif
		if (n->syn1 == NULL) {Rprintf("Memory allocation failed\n"); exit(1);}
		for (b = 0; b < o->layer1_size; b++) for (a = 0; a < v->vocab_size; a++)
			n->syn1[a * o->layer1_size + b] = 0;
	}
	if (o->negative>0) {
#ifdef _WIN32
		n->syn1neg = (real *)malloc((long long)v->vocab_size * o->layer1_size * sizeof(real));
#elif __posix
		a = posix_memalign((void **)&(n->syn1neg), 128, (long long)v->vocab_size * o->layer1_size * sizeof(real));
#endif
		if (n->syn1neg == NULL) {Rprintf("Memory allocation failed\n"); exit(1);}
		for (b = 0; b < o->layer1_size; b++) for (a = 0; a < v->vocab_size; a++)
			n->syn1neg[a * o->layer1_size + b] = 0;
	}
	for (b = 0; b < o->layer1_size; b++) for (a = 0; a < v->vocab_size; a++)
		n->syn0[a * o->layer1_size + b] = (rand() / (real)RAND_MAX - 0.5) / o->layer1_size;
	CreateBinaryTree(v);
}

static void TrainModelThread(struct all_data *global_data) {
	Rprintf("Training model...\n");
	struct all_data *data = global_data;
	long long id = data->id;
	struct net *n = data->n;
	struct vocab * v = data->v;
	int *table = data->table;
	struct configs *o = data->o;

	long long a, b, d, word, last_word, sentence_length = 0, sentence_position = 0;
	long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
	long long l1, l2, c, target, label;
	unsigned long long next_random = (long long)id;
	real f, g;
	clock_t now;
	real *neu1 = (real *)calloc(o->layer1_size, sizeof(real));
	real *neu1e = (real *)calloc(o->layer1_size, sizeof(real));
	FILE *fi = fopen(o->train_file, "rb");
	fseek(fi, o->file_size / (long long)o->num_threads * (long long)id, SEEK_SET);
	while (1) {
		if (word_count - last_word_count > 10000) {
			o->word_count_actual += word_count - last_word_count;
			last_word_count = word_count;
			if ((o->debug_mode > 1)) {
				now=clock();
				Rprintf("%cAlpha: %f  Progress: %.2f%%  Words/sec: %.2fk  ", 13, o->alpha,
						o->word_count_actual / (real)(o->train_words + 1) * 100,
						o->word_count_actual / ((real)(now - o->start + 1) / (real)CLOCKS_PER_SEC * 1000));
				/*fflush(stdout);*/
			}
			o->alpha = o->starting_alpha * (1 - o->word_count_actual / (real)(o->train_words + 1));
			if (o->alpha < o->starting_alpha * 0.0001) o->alpha = o->starting_alpha * 0.0001;
		}
		if (sentence_length == 0) {
			while (1) {
				word = ReadWordIndex(fi,v);
				if (feof(fi)) break;
				if (word == -1) continue;
				word_count++;
				if (word == 0) break;
				// The subsampling randomly discards frequent words while keeping the ranking same
				if (o->sample > 0) {
					real ran = (sqrt(v->vocab[word].cn / (o->sample * o->train_words)) + 1) * (o->sample * o->train_words) / v->vocab[word].cn;
					next_random = next_random * (unsigned long long)25214903917 + 11;
					if (ran < (next_random & 0xFFFF) / (real)65536) continue;
				}
				sen[sentence_length] = word;
				sentence_length++;
				if (sentence_length >= MAX_SENTENCE_LENGTH) break;
			}
			sentence_position = 0;
		}
		if (feof(fi)) break;
		if (word_count > o->train_words / o->num_threads) break;
		word = sen[sentence_position];
		if (word == -1) continue;
		for (c = 0; c < o->layer1_size; c++) neu1[c] = 0;
		for (c = 0; c < o->layer1_size; c++) neu1e[c] = 0;
		next_random = next_random * (unsigned long long)25214903917 + 11;
		b = next_random % o->window;
		if (o->cbow) {  //train the cbow architecture
			// in -> hidden
			for (a = b; a < o->window * 2 + 1 - b; a++) if (a != o->window) {
				c = sentence_position - o->window + a;
				if (c < 0) continue;
				if (c >= sentence_length) continue;
				last_word = sen[c];
				if (last_word == -1) continue;
				for (c = 0; c < o->layer1_size; c++) neu1[c] += n->syn0[c + last_word * o->layer1_size];
			}
			if (o->hs) for (d = 0; d < v->vocab[word].codelen; d++) {
				f = 0;
				l2 = v->vocab[word].point[d] * o->layer1_size;
				// Propagate hidden -> output
				for (c = 0; c < o->layer1_size; c++) f += neu1[c] * n->syn1[c + l2];
				if (f <= -MAX_EXP) continue;
				else if (f >= MAX_EXP) continue;
				else f = n->expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
				// 'g' is the gradient multiplied by the learning rate
				g = (1 - v->vocab[word].code[d] - f) * o->alpha;
				// Propagate errors output -> hidden
				for (c = 0; c < o->layer1_size; c++) neu1e[c] += g * n->syn1[c + l2];
				// Learn weights hidden -> output
				for (c = 0; c < o->layer1_size; c++) n->syn1[c + l2] += g * neu1[c];
			}
			// NEGATIVE SAMPLING
			if (o->negative > 0) for (d = 0; d < o->negative + 1; d++) {
				if (d == 0) {
					target = word;
					label = 1;
				} else {
					next_random = next_random * (unsigned long long)25214903917 + 11;
					target = table[(next_random >> 16) % TABLE_SIZE];
					if (target == 0) target = next_random % (v->vocab_size - 1) + 1;
					if (target == word) continue;
					label = 0;
				}
				l2 = target * o->layer1_size;
				f = 0;
				for (c = 0; c < o->layer1_size; c++) f += neu1[c] * n->syn1neg[c + l2];
				if (f > MAX_EXP) g = (label - 1) * o->alpha;
				else if (f < -MAX_EXP) g = (label - 0) * o->alpha;
				else g = (label - n->expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * o->alpha;
				for (c = 0; c < o->layer1_size; c++) neu1e[c] += g * n->syn1neg[c + l2];
				for (c = 0; c < o->layer1_size; c++) n->syn1neg[c + l2] += g * neu1[c];
			}
			// hidden -> in
			for (a = b; a < o->window * 2 + 1 - b; a++) if (a != o->window) {
				c = sentence_position - o->window + a;
				if (c < 0) continue;
				if (c >= sentence_length) continue;
				last_word = sen[c];
				if (last_word == -1) continue;
				for (c = 0; c < o->layer1_size; c++) n->syn0[c + last_word * o->layer1_size] += neu1e[c];
			}
		} else {  //train skip-gram
			for (a = b; a < o->window * 2 + 1 - b; a++) if (a != o->window) {
				c = sentence_position - o->window + a;
				if (c < 0) continue;
				if (c >= sentence_length) continue;
				last_word = sen[c];
				if (last_word == -1) continue;
				l1 = last_word * o->layer1_size;
				for (c = 0; c < o->layer1_size; c++) neu1e[c] = 0;
				// HIERARCHICAL SOFTMAX
				if (o->hs) for (d = 0; d < v->vocab[word].codelen; d++) {
					f = 0;
					l2 = v->vocab[word].point[d] * o->layer1_size;
					// Propagate hidden -> output
					for (c = 0; c < o->layer1_size; c++) f += n->syn0[c + l1] * n->syn1[c + l2];
					if (f <= -MAX_EXP) continue;
					else if (f >= MAX_EXP) continue;
					else f = n->expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
					// 'g' is the gradient multiplied by the learning rate
					g = (1 - v->vocab[word].code[d] - f) * o->alpha;
					// Propagate errors output -> hidden
					for (c = 0; c < o->layer1_size; c++) neu1e[c] += g * n->syn1[c + l2];
					// Learn weights hidden -> output
					for (c = 0; c < o->layer1_size; c++) n->syn1[c + l2] += g * n->syn0[c + l1];
				}
				// NEGATIVE SAMPLING
				if (o->negative > 0) for (d = 0; d < o->negative + 1; d++) {
					if (d == 0) {
						target = word;
						label = 1;
					} else {
						next_random = next_random * (unsigned long long)25214903917 + 11;
						target = table[(next_random >> 16) % TABLE_SIZE];
						if (target == 0) target = next_random % (v->vocab_size - 1) + 1;
						if (target == word) continue;
						label = 0;
					}
					l2 = target * o->layer1_size;
					f = 0;
					for (c = 0; c < o->layer1_size; c++) f += n->syn0[c + l1] * n->syn1neg[c + l2];
					if (f > MAX_EXP) g = (label - 1) * o->alpha;
					else if (f < -MAX_EXP) g = (label - 0) * o->alpha;
					else g = (label - n->expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * o->alpha;
					for (c = 0; c < o->layer1_size; c++) neu1e[c] += g * n->syn1neg[c + l2];
					for (c = 0; c < o->layer1_size; c++) n->syn1neg[c + l2] += g * n->syn0[c + l1];
				}
				// Learn weights input -> hidden
				for (c = 0; c < o->layer1_size; c++) n->syn0[c + l1] += neu1e[c];
			}
		}
		sentence_position++;
		if (sentence_position >= sentence_length) {
			sentence_length = 0;
			continue;
		}
	}
	fclose(fi);
	free(neu1);
	free(neu1e);
	/*pthread_exit(NULL);*/
}

static void TrainModel(struct vocab *v, struct net *n, int *table, struct configs *o) {
	long a, b, c, d;
	FILE *fo;
	/*pthread_t *pt = (pthread_t *)malloc(o->num_threads * sizeof(pthread_t));*/
	Rprintf("Starting training using file %s\n", o->train_file);
	o->starting_alpha = o->alpha;
	if (o->read_vocab_file[0] != 0)
		ReadVocab(v, o);
	else {
		LearnVocabFromTrainFile(v, o);
	}
	if (o->save_vocab_file[0] != 0) SaveVocab(v, o);
	if (o->output_file[0] == 0) return;
	InitNet(n, v, o);
	if (o->negative > 0) InitUnigramTable(table, v);
	o->start = clock();
	for (a = 0; a < o->num_threads; a++) {
		struct all_data *data = malloc(sizeof(struct all_data));
		data->v = v;
		data->n = n;
		data->table = table;
		data->o = o;
		data->id = a;
		/*pthread_create(&pt[a], NULL, TrainModelThread, (void *)data);*/
		TrainModelThread(data);
	}
	/*for (a = 0; a < o->num_threads; a++) pthread_join(pt[a], NULL);*/
	fo = fopen(o->output_file, "wb");
	if (o->classes == 0) {
		// Save the word vectors
		fprintf(fo, "%lld %lld\n", v->vocab_size, o->layer1_size);
		for (a = 0; a < v->vocab_size; a++) {
			fprintf(fo, "%s ", v->vocab[a].word);
			if (o->binary) for (b = 0; b < o->layer1_size; b++) fwrite(&(n->syn0[a * o->layer1_size + b]), sizeof(real), 1, fo);
			else for (b = 0; b < o->layer1_size; b++) fprintf(fo, "%lf ", n->syn0[a * o->layer1_size + b]);
			fprintf(fo, "\n");
		}
	} else {
		// Run K-means on the word vectors
		int clcn = o->classes, iter = 10, closeid;
		int *centcn = (int *)malloc(o->classes * sizeof(int));
		int *cl = (int *)calloc(v->vocab_size, sizeof(int));
		real closev, x;
		real *cent = (real *)calloc(o->classes * o->layer1_size, sizeof(real));
		for (a = 0; a < v->vocab_size; a++) cl[a] = a % clcn;
		for (a = 0; a < iter; a++) {
			for (b = 0; b < clcn * o->layer1_size; b++) cent[b] = 0;
			for (b = 0; b < clcn; b++) centcn[b] = 1;
			for (c = 0; c < v->vocab_size; c++) {
				for (d = 0; d < o->layer1_size; d++) cent[o->layer1_size * cl[c] + d] += n->syn0[c * o->layer1_size + d];
				centcn[cl[c]]++;
			}
			for (b = 0; b < clcn; b++) {
				closev = 0;
				for (c = 0; c < o->layer1_size; c++) {
					cent[o->layer1_size * b + c] /= centcn[b];
					closev += cent[o->layer1_size * b + c] * cent[o->layer1_size * b + c];
				}
				closev = sqrt(closev);
				for (c = 0; c < o->layer1_size; c++) cent[o->layer1_size * b + c] /= closev;
			}
			for (c = 0; c < v->vocab_size; c++) {
				closev = -10;
				closeid = 0;
				for (d = 0; d < clcn; d++) {
					x = 0;
					for (b = 0; b < o->layer1_size; b++) x += cent[o->layer1_size * d + b] * n->syn0[c * o->layer1_size + b];
					if (x > closev) {
						closev = x;
						closeid = d;
					}
				}
				cl[c] = closeid;
			}
		}
		// Save the K-means classes
		for (a = 0; a < v->vocab_size; a++) fprintf(fo, "%s %d\n", v->vocab[a].word, cl[a]);
		free(centcn);
		free(cent);
		free(cl);
	}
	fclose(fo);
}

SEXP load_vectors(SEXP filepath) {
	FILE *f;
	long long a, b;
	float len;
	char ch;
	struct internal_data data;
	SEXP names, vectors;

	f = fopen(CHAR(STRING_ELT(filepath, 0)), "rb");
	if (f == NULL) {
		error("Input file not found\n");
	}
	fscanf(f, "%lld", &(data.words));
	fscanf(f, "%lld", &(data.size));
	data.vocab = (char *)malloc((long long)(data.words) * MAX_W * sizeof(char));
	data.M = (float *)malloc((long long)(data.words) * (long long)(data.size) * sizeof(float));
	if (data.M == NULL) {
		Rprintf("Cannot allocate memory: %lld MB    %lld  %lld\n", (long long)(data.words) * data.size * sizeof(float) / 1048576, data.words, data.size);
		error("Failed to allocate memory");
	}

	names = PROTECT(allocVector(STRSXP, data.words));
	vectors = PROTECT(allocVector(VECSXP, data.words));

	for (b = 0; b < data.words; b++) {
		SEXP vector = PROTECT(allocVector(REALSXP, data.size));

		fscanf(f, "%s%c", &(data.vocab)[b * MAX_W], &ch);
		SET_STRING_ELT(names, b, mkChar(&data.vocab[b*MAX_W]));
		for (a = 0; a < data.size; a++) fread(&(data.M)[a + b * data.size], sizeof(float), 1, f);
		len = 0;
		for (a = 0; a < data.size; a++) len += data.M[a + b * data.size] * data.M[a + b * data.size];
		len = sqrt(len);
		for (a = 0; a < data.size; a++) {
			data.M[a + b * data.size] /= len;
			REAL(vector)[a] = data.M[a + b * data.size];
		}

		SET_VECTOR_ELT(vectors, b, vector);
		UNPROTECT(1);
	}
	fclose(f);

	setAttrib(vectors, R_NamesSymbol, names);
	UNPROTECT(2);

	free(data.vocab);
	free(data.M);
	return vectors;
}

static int train_model_with_config(struct configs *o) {
	struct vocab *v = malloc(sizeof(struct vocab));
	InitVocab(v);

	struct net *n = malloc(sizeof(struct net));
	InitNetData(n);

	int *table = NULL;
	TrainModel(v, n, table, o);
	return 0;
}

static int train_model(const char* input, const char* output) {
	struct configs *o = malloc(sizeof(struct configs));
	InitConfigs(o);
	strcpy(o->train_file, input);
	strcpy(o->output_file, output);

	train_model_with_config(o);
	return 0;
}

SEXP train(SEXP input, SEXP output) {
	SEXP res;
	PROTECT(res = allocVector(INTSXP, 1));
	train_model(CHAR(STRING_ELT(input, 0)), CHAR(STRING_ELT(output, 0)));
	INTEGER(res)[0] = 0;
	UNPROTECT(1);
	return(res);
}

/*int main() {*/
	/*train_model("text8", "vectors_test.bin");*/
	/*return 0;*/
/*}*/
