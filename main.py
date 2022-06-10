import random
import math
from matplotlib import pyplot as plt


class GeneticAlgorithm:
    def __init__(self, chromosome, max_number_of_chromosomes=100, mutation_rate=0.1, decay=10,
                 elite_selection=0.01, epochs=200):
        # having an interface between this and the class the user calls means the user does not have to specify all the values, as there can be default values
        self.chromosomes = [chromosome]  # chromosome
        self.chromosome_length = len(chromosome)
        self.number_of_chromosomes = 1
        self.max_number_of_chromosomes = max_number_of_chromosomes

        self.mutation_rate = mutation_rate
        self.decay = decay
        self.elite_selection = elite_selection
        self.highest_scoring_chromosomes = []
        self.epochs = epochs

        self.chromosome_scores = []
        self.generation_scores = []
        self.best_generation = [0, 0, 0]  # [generation, chromosomes, generation score]
        self.best_chromosome = [0, 0, 0]  # [generation, chromosome, score]

    def evolve(self):
        for i in range(self.epochs):
            new_generation = []
            self.high_scoring_chromosomes()

            elite = self.elite()
            mutations = self.mutate(self.highest_scoring_chromosomes,
                                    number_of_mutations=int(self.chromosome_length * self.mutation_rate / self.decay))
            crossover = self.crossover(self.highest_scoring_chromosomes)

            new_generation.extend(elite)
            new_generation.extend(mutations)
            new_generation.extend(crossover)

            self.chromosomes = new_generation
            self.number_of_chromosomes = len(self.chromosomes)
            self.score_chromosomes()
            self.remove_weakest_chromosomes()

            if sum(elite[0]) / self.chromosome_length > self.best_chromosome[2]:
                self.best_chromosome = [i, elite[0], sum(elite[0]) / self.chromosome_length]

            self.generation_scores.append(self.score_generation())
            self.high_scoring_chromosomes()

            if self.best_generation[2] < self.generation_scores[-1]:
                self.best_generation = [i, self.chromosomes, self.generation_scores[-1]]

    def remove_weakest_chromosomes(self):
        for i in range(self.number_of_chromosomes - self.max_number_of_chromosomes):
            lowest_scored_chromosome = {"index": 0, "score": self.chromosome_scores[0]}
            for index, chromosome_score in enumerate(self.chromosome_scores):
                if chromosome_score < lowest_scored_chromosome["score"]:
                    lowest_scored_chromosome["score"] = chromosome_score
            del self.chromosomes[lowest_scored_chromosome["index"]]
            del self.chromosome_scores[lowest_scored_chromosome["index"]]
        self.number_of_chromosomes = len(self.chromosomes)

    def elite(self):
        number_of_elites = int(self.number_of_chromosomes * self.elite_selection) + 10
        elite = []

        for chromosome in self.chromosomes:
            score = self.score(chromosome)
            if len(elite) != number_of_elites:
                elite.append([chromosome, score])
            elif score > elite[-1][1]:
                elite[-1] = [chromosome, score]

            elite.sort(key=lambda row: row[1], reverse=True)

        for index, chromosome in enumerate(elite):
            elite[index] = chromosome[0]

        return elite

    def mutate(self, chromosomes, number_of_mutations):
        mutations = []
        for i in range(len(chromosomes)):
            mutation = list(chromosomes[i])
            for k in range(number_of_mutations):
                gene = random.randint(0, self.chromosome_length - 1)
                mutation[gene] = (chromosomes[i][gene] + 1) % 2
            mutations.append(mutation)

        return mutations

    def crossover(self, elite):
        chromosomes = []
        for chromosome1 in elite:
            for chromosome2 in elite:
                if chromosome1 != chromosome2:
                    new_chromosome1, new_chromosome2 = self.cross(chromosome1, chromosome2, number_of_cross_points=2)
                    chromosomes.append(new_chromosome1)
                    chromosomes.append(new_chromosome2)
            if len(chromosomes) > self.max_number_of_chromosomes:
                break
        return chromosomes

    def cross(self, chromosome1, chromosome2, number_of_cross_points):
        cross_points = self.generate_cross_points(number_of_cross_points)

        new_chromosome1 = []
        new_chromosome2 = []

        for i in range(number_of_cross_points + 1):
            if i % 2:
                new_chromosome1.extend(chromosome1[cross_points[i]:cross_points[i + 1]])
                new_chromosome2.extend(chromosome2[cross_points[i]:cross_points[i + 1]])
            else:
                new_chromosome1.extend(chromosome2[cross_points[i]:cross_points[i + 1]])
                new_chromosome2.extend(chromosome1[cross_points[i]:cross_points[i + 1]])

        return new_chromosome1, new_chromosome2

    def generate_cross_points(self, number_of_cross_points):
        cross_points = [0]
        cross_points.extend([random.randint(1, self.chromosome_length - 1) for i in range(number_of_cross_points)])
        cross_points.append(self.chromosome_length)
        return sorted(cross_points)

    def high_scoring_chromosomes(self):
        highest_scoring_chromosomes = []
        chromosome_indexes = []
        high_scoring_selection = int(self.number_of_chromosomes / 5) + 10
        for i in range(high_scoring_selection):
            if i >= len(self.chromosome_scores):
                self.highest_scoring_chromosomes = self.chromosomes
                return

            best_score = 0
            chromosome_index = 0

            for index, chromosome_score in enumerate(self.chromosome_scores):
                if index in chromosome_indexes:
                    continue
                if chromosome_score > best_score:
                    best_score = chromosome_score
                    chromosome_index = index

            if chromosome_index in chromosome_indexes:
                continue
            chromosome_indexes.append(chromosome_index)
            highest_scoring_chromosomes.append(self.chromosomes[chromosome_index])
        self.highest_scoring_chromosomes = highest_scoring_chromosomes
        return

    def score(self, chromosome):
        score = sum(chromosome)
        return score

    def score_generation(self):
        score = 0
        for chromosome in self.chromosomes:
            score += sum(chromosome)
        return round((score / len(self.chromosomes)) / self.chromosome_length, 2)

    def score_chromosomes(self):
        self.chromosome_scores = []
        for chromosome in self.chromosomes:
            self.chromosome_scores.append(self.score(chromosome))


def generate_bit_string(length):
    return [random.randint(0, 1) for i in range(length)]


def plot_data(specie, generations):
    x = [i for i in range(generations)]

    plt.plot(x, specie.generation_scores, color='blue')
    plt.title("Genetic Algorithm")
    plt.xlabel("Generations")
    plt.ylabel("Fitness Score")
    plt.savefig('GeneticAlgorithm2.png')
    plt.show()


def print_results(specie):
    print(
        f'The best generation was generation {specie.best_generation[0]}, with an average score of: {specie.best_generation[2]}')
    print(f'Number of chromosomes in the generation: {len(specie.best_generation[1])}')
    print('')
    print(
        f'The best chromosome was from generation: {specie.best_chromosome[0]}, with a score of: {specie.best_chromosome[2]}, Genome sequence: {specie.best_chromosome[1]}')


def main():
    bit_string = generate_bit_string(length=100)
    epochs = 200
    specie = GeneticAlgorithm(chromosome=bit_string, epochs=epochs)
    specie.evolve()
    print_results(specie)

    plot_data(specie, epochs)


if __name__ == "__main__":
    main()
