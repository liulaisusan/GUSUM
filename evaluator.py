import rouge
from tqdm import tqdm

class Evaluator:
    def __init__(self, summarizers):
        self.summarizers = summarizers
        self.all_corpus = []
        self.all_goldensum = []
        self.all_gusum = []
        self.all_hybridsum = []
        self.all_modelsum = []
    
    def evaluate(self, dataset, num_of_documents = None, aggregators=['Avg'], print_results = True):
        if num_of_documents:
            N = num_of_documents
        else:
            N = len(dataset)
        for d in tqdm(range(N)):
            if d % 1000 == 0:
                self._allRougeEvaluation(aggregators)
            corpus=dataset['dialogue'][d]
            self.all_corpus.append(corpus)
            golden_summary=dataset['summary'][d]
            self.all_goldensum.append(golden_summary)
            gusum_sum = None
            hybrid_sum = None
            model_sum = None
            for summarizer in self.summarizers:
                if summarizer.name == 'gusum':
                    gusum_sum = summarizer.summarize(corpus)
                    self.all_gusum.append(gusum_sum)
                elif summarizer.name == 'hybrid':
                    if gusum_sum:
                        hybrid_sum = summarizer.summarize(corpus, gusum_sum)
                    else:
                        hybrid_sum = summarizer.summarize(corpus)
                    self.all_hybridsum.append(hybrid_sum)
                elif summarizer.name == 'model':
                    model_sum = summarizer.summarize(corpus)
                    self.all_modelsum.append(model_sum)
            # if print_results:
            #     self.printSingleSummary(d, corpus, golden_summary, gusum_sum, hybrid_sum, model_sum)
        if print_results:
            self.printAllSummaries()
        self._allRougeEvaluation(aggregators)

    def printSingleSummary(self, d, corpus = None, golden_sum = None, gusum_sum = None, hybrid_sum = None, model_sum = None):
        print("="*100)
        if corpus:
            print(f"----- Dialog {d} ----- \n {corpus}")
        if golden_sum:
            print(f"----- Golden Summary {d} -----\n {golden_sum}")
        if gusum_sum:
            print(f"----- Gusum Summary {d} ----- \n {gusum_sum}")
        if hybrid_sum:
            print(f"----- Hybrid Summary {d} ----- \n {hybrid_sum}")
        if model_sum:
            print(f"----- Model Summary {d} ----- \n {model_sum}")
    
    def printAllSummaries(self):
        is_gusum = True if len(self.all_gusum) != 0 else False
        is_hybrid = True if len(self.all_hybridsum) != 0 else False
        is_model = True if len(self.all_modelsum) != 0 else False

        for d in range(len(self.all_corpus)):
            gusum = self.all_gusum[d] if is_gusum else None
            hybrid = self.all_hybridsum[d] if is_hybrid else None
            model = self.all_modelsum[d] if is_model else None
            self.printSingleSummary(d, self.all_corpus[d], self.all_goldensum[d], gusum, hybrid, model)

    def _allRougeEvaluation(self, aggregators=['Avg']):
        tqdm.write("="*100)
        if len(self.all_gusum) != 0:
            self.rougeEvaluation('Gusum', self.all_gusum, self.all_goldensum, aggregators)
        if len(self.all_hybridsum) != 0:
            self.rougeEvaluation('Hybrid', self.all_hybridsum, self.all_goldensum, aggregators)
        if len(self.all_modelsum) != 0:
            self.rougeEvaluation('Model', self.all_modelsum, self.all_goldensum, aggregators)      

    def prepare_results(self, m, p, r, f):
        return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(m, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)

    def rougeEvaluation(self, sum_name, all_hypothesis, all_references, aggregators=['Avg']):

        for aggregator in aggregators:
            tqdm.write(f'Evaluation {sum_name} with {sum_name}'.format(sum_name, aggregator))
            apply_avg = aggregator == 'Avg'
            apply_best = aggregator == 'Best'
        
            evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                                max_n=4,
                                limit_length=False,
                                length_limit=1000,
                                length_limit_type='words',
                                apply_avg=apply_avg,
                                apply_best=apply_best,
                                alpha=0.2, # Default F1_score
                                weight_factor=1.2,
                                stemming=True)
        
            scores = evaluator.get_scores(all_hypothesis, all_references)
        
            for metric, results in sorted(scores.items(), key=lambda x: x[0]):
                if not apply_avg and not apply_best: # value is a type of list as we evaluate each summary vs each reference
                    for hypothesis_id, results_per_ref in enumerate(results):
                        nb_references = len(results_per_ref['p'])
                        for reference_id in range(nb_references):
                            tqdm.write('\tHypothesis #{} & Reference #{}: '.format(hypothesis_id, reference_id))
                            tqdm.write('\t' + self.prepare_results(metric,results_per_ref['p'][reference_id], results_per_ref['r'][reference_id], results_per_ref['f'][reference_id]))
                    tqdm.write('\n')
                else:
                    tqdm.write(self.prepare_results(metric, results['p'], results['r'], results['f']))
            tqdm.write('\n')