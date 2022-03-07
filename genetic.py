import random
import time

# Constantes de Position dans le tuple
X=0;    # X
Y=1;    # f(X)
P=2;    # Probabilité / Poids
C=3;    # Vie en nombre de cycles

class Utils:

    @staticmethod
    def binaire( x: 'int', bit_nombre: 'int' = 8 ) -> 'list':
        """Donne l'écriture binaire sur `bit_nombre` bits."""
        tableau = [None] * bit_nombre
        i = 0
        while i < bit_nombre:
            tableau[i] = bool(x & 1)
            x >>= 1
            i += 1 
        return tableau

    @staticmethod
    def entier( tableau: 'list' ) -> int:
        """Converti l'écriture binaire en nombre entier."""
        l = len(tableau) - 1
        r = 0
        while l >= 0:
            r = (r << 1) + tableau[l]
            l -= 1
        return r

class Tri:

    @staticmethod
    def fusionInterne( partie_gauche: 'list', partie_droite: 'list', compare_fn=lambda x, y: x < y ) -> 'list':
            """Tri Fusion - Fonction Interne"""
            longueur_gauche = len( partie_gauche )
            longueur_droite = len( partie_droite )
            i = 0
            j = 0
            sortie = [None] * ( longueur_gauche + longueur_droite )
            while i < longueur_gauche and j < longueur_droite:
                if compare_fn( partie_gauche[i], partie_droite[j] ):
                    sortie[i+j] = partie_gauche[i]
                    i += 1
                else:
                    sortie[i+j] = partie_droite[j]
                    j += 1

            while i < longueur_gauche:
                sortie[i+j] = partie_gauche[i]
                i += 1

            while j < longueur_droite:
                sortie[i+j] = partie_droite[j]
                j += 1

            return sortie

    @staticmethod
    def fusion( tableau: 'list', compare_fn=lambda x, y: x < y ) -> 'list':
        """Tri Fusion"""
        if len( tableau ) > 1:
            middle = len( tableau ) // 2
            partie_gauche = tableau[:middle]
            partie_droite = tableau[middle:]
            return Tri.fusionInterne(
                Tri.fusion( partie_gauche, compare_fn ),
                Tri.fusion( partie_droite, compare_fn ),
                compare_fn
            )
        else:
            return tableau

class DefaultGenetique:

    __core_crossover__ = {
        'min_ratio': 0.25,
        'max_ratio': 0.75,
        'fixed'    : False,
    }

    __core_mutation__ = {
        'ratio'    : 0.01,
        'fixed'    : False,
    }

    @staticmethod
    def core_joint( core: 'dict' ) -> 'dict':
        return {
            'crossover': { **DefaultGenetique.__core_crossover__, **core.get( 'crossover', {} ) },
            'mutation' : { **DefaultGenetique.__core_mutation__, **core.get( 'mutation', {} ) },
        }

    @staticmethod
    def __compare_fn__( x: 'tuple', y: 'tuple' ) -> 'bool':
        return x[Y] < y[Y]

    @staticmethod
    def __pre_fn__( x: 'tuple', contexte: 'Genetique' ) -> 'any':
        return x

    @staticmethod
    def __post_fn__( x: 'tuple', contexte: 'Genetique' ) -> 'any':
        return x

    @staticmethod
    def __init_fn__( contexte: 'Genetique' = None ) -> 'any':
        return random.random()

    @staticmethod
    def __fitness_fn__( x: 'float', contexte: 'Genetique' = None ) -> 'float':
        return x

    @staticmethod
    def __fitness_somme_fn__( population: 'list', contexte: 'Genetique' = None ) -> 'float|int':
        somme = 0
        for x in population:
            somme += x[Y]
        return somme

    @staticmethod
    def __proba_fn__( population: 'list', contexte: 'Genetique' = None ) -> 'float':
        somme = DefaultGenetique.__fitness_somme_fn__( population, contexte )
        for i in range(0,len(population)):
            population[i] = (population[i][X],population[i][Y],population[i][Y] / somme,population[i][C])
        return somme

    @staticmethod
    def __encoder_fn__( x: 'tuple', contexte: 'Genetique' = None ) -> 'any':
        return x
    
    @staticmethod
    def __decoder_fn__( x: 'tuple', contexte: 'Genetique' = None ) -> 'any':
        return x

    @staticmethod
    def __mutate_fn__( x: 'any', contexte: 'Genetique' = None ) -> 'any':
        return x

    @staticmethod
    def __xover_fn__( x: 'any', y: 'any', contexte: 'Genetique' = None ) -> 'list':
        return [x, y]

    @staticmethod
    def __valid_fn__( contexte: 'Genetique' = None ) -> 'bool':
        return True

def new_tuple( x: 'float', contexte: 'Genetique' ) -> 'tuple':
    return ( contexte.pre_fn( x, contexte ), contexte.fitness_fn( x, contexte ), 0, 0 )

class Genetique( DefaultGenetique ):

    @staticmethod
    def roulette( population: 'list', somme: 'float' ) -> 'list':
        while True:
            aleatoire = random.random() * somme
            for i in range(0,len(population)):
                aleatoire -= population[i][P]
                if aleatoire <= 0:
                    return population[i]

    def __init__(
        self,
        taille:     'int'       = 100,
        expiration: 'int'       = 3,
        parametres: 'dict'      = {},
        core:       'dict'      = {},
        init_fn :   'callable'  = DefaultGenetique.__init_fn__,
        fitness_fn: 'callable'  = DefaultGenetique.__fitness_fn__,
        proba_fn:   'callable'  = DefaultGenetique.__proba_fn__,
        pre_fn:     'callable'  = DefaultGenetique.__pre_fn__,
        post_fn:    'callable'  = DefaultGenetique.__post_fn__,
        encoder_fn: 'callable'  = DefaultGenetique.__encoder_fn__,
        decoder_fn: 'callable'  = DefaultGenetique.__decoder_fn__,
        mutate_fn:  'callable'  = DefaultGenetique.__mutate_fn__,
        xover_fn:   'callable'  = DefaultGenetique.__xover_fn__,
        compare_fn: 'callable'  = DefaultGenetique.__compare_fn__,
        valid_fn:   'callable'  = DefaultGenetique.__valid_fn__,
    ):
        self.taille         = taille
        self.expiration     = expiration
        self.parametres     = parametres
        self.core           = DefaultGenetique.core_joint( core )
        
        self.init_fn        = init_fn
        self.fitness_fn     = fitness_fn
        self.proba_fn       = proba_fn
        self.pre_fn         = pre_fn
        self.post_fn        = post_fn
        self.encoder_fn     = encoder_fn
        self.decoder_fn     = decoder_fn
        self.mutate_fn      = mutate_fn
        self.xover_fn       = xover_fn
        self.compare_fn     = compare_fn
        self.valid_fn       = valid_fn
        
        self.ensembles = [ self.initialiser_population() ]
    
    def trier_population( self, population: 'list' ) -> 'list':
        return Tri.fusion( population, self.compare_fn )

    def initialiser_population( self ) -> 'list':
        population = [None] * self.taille
        for i in range( 0, self.taille ):
            population[i] = new_tuple( self.init_fn( self ), self )
        return self.trier_population( population )

    def calculer_proba( self, population: 'list' ) -> 'float|int':
        return self.proba_fn( population, self )

    def encode( self, x: 'tuple' ) -> 'any':
        return self.encoder_fn( x, self )
    
    def decode( self, x: 'any' ) -> 'any':
        return self.decoder_fn( x, self )

    def mutate( self, x: 'any' ) -> 'int':
        return self.mutate_fn( x, self )

    def crossover( self, parent1: 'any', parent2: 'any' ) -> 'list':
        return self.xover_fn( parent1, parent2, self )

    def selectionner_parents( self, population: 'list' ) -> 'list':
        somme = self.calculer_proba( population )     
        min_ratio = self.core['crossover'].get( 'min_ratio', 0.25 )
        max_ratio = self.core['crossover'].get( 'max_ratio', 0.75 )
        fixed     = self.core['crossover'].get( 'fixed', False )
        if type(fixed) == int and fixed > 0:
            taille = fixed
            parents = [None] * taille * 2
            for i in range( 0, taille ):
                j = i * 2
                k = j + 1
                parents[j] = Genetique.roulette( population, somme )
                while parents[k] == None or parents[j] == parents[k]:
                    parents[k] = Genetique.roulette( population, somme )
            return parents
        else:
            ratio = ( random.random() * ( max_ratio - min_ratio ) ) + min_ratio
            length = int( len( population ) * ratio )
            length = length if length > 2 else 2
            length = length if length % 2 == 0 else length + 1
            parents = [None] * length
            for i in range( 0, length // 2 ):
                j = i * 2
                k = j + 1
                parents[j] = Genetique.roulette( population, somme )
                while parents[k] == None or parents[j] == parents[k]:
                    parents[k] = Genetique.roulette( population, somme )
            return parents

    def get_enfants( self, population: 'list' ) -> 'list':
        mutate_ratio = self.core['mutation'].get( 'ratio', 0.01 )
        mutate_fixed = self.core['mutation'].get( 'fixed', False )
        enfants = []
        parents = self.selectionner_parents( population )
        for i in range( 0, len( parents ) // 2 ):
            for enfant in self.crossover( parents[i*2], parents[i*2+1] ):
                new_enfant = enfant
                if type(mutate_fixed) == int and mutate_fixed > 0:
                    new_enfant = self.mutate( enfant )
                    mutate_fixed -= 1
                elif random.random() < mutate_ratio:
                    new_enfant = self.mutate( enfant )
                enfants.append( self.decode( new_enfant ) )
        return enfants

    def compute( self, garder: 'bool' = False ) -> 'list':
        population = self.dernierEnsemble()
        if len( population ) > 2:
            enfants = self.get_enfants( population )
            sortie = [ new_tuple( self.post_fn( x, self ), self ) for x in enfants ]
            if ( garder == True ):
                for i in range(0,len(population)):
                    if population[i][C] + 1 < self.expiration:
                        sortie.append( ( population[i][X], population[i][Y], population[i][P], population[i][C] ) )
            return self.ajouterEnsemble( self.trier_population( sortie ) )
        else:
            return population

    def ajouterEnsemble( self, set: 'list' ) -> 'bool':
        self.ensembles.append( set )
        return set if self.ensembles[-1] == set else Exception( "Set has not been added" )

    def dernierEnsemble( self ) -> 'list':
        return self.ensembles[-1]

    def meilleur( self ) -> 'float':
        return self.post_fn( self.dernierEnsemble()[0][X], self )

    def valid( self ) -> 'bool':
        return self.valid_fn( self )

    def run( self, iteration: 'int' = 1, garder: 'bool' = False ) -> 'float':
        for _ in range( 0, iteration ):
            self.compute( garder )
        return self.meilleur()

def TP_Projet_IA():

    def _tp_fitness_fn( x: 'float', contexte: 'Genetique' ):
        """Fonction ... : 2 * ( x ** 2 ) - x"""
        return 2 * ( x ** 2 ) - x

    def _tp_pre_fn( x: 'float', contexte: 'Genetique' ) -> 'int':
        return int( x * contexte.parametres['puissance'] )

    def _tp_encoder_fn( x: 'tuple', contexte: 'Genetique' ) -> 'list':
        return Utils.binaire( x[X], contexte.parametres['bit_nombre'] )

    def _tp_xover_fn( parent1: 'tuple', parent2: 'tuple', contexte: 'Genetique' ) -> 'list':
        tableau1 = contexte.encode( parent1 )
        tableau2 = contexte.encode( parent2 )
        min_taille = min( [ len( tableau1 ), len( tableau2 ) ] )
        min_xover_taille = int( min_taille * contexte.parametres['min_xover_ratio'] )
        max_xover_taille = int( min_taille * contexte.parametres['max_xover_ratio'] )
        xover_taille = random.randint( min_xover_taille, max_xover_taille )
        xover_debut = random.randint( 0, min_taille - xover_taille )
        xover_fin = xover_debut + xover_taille
        enfant1 = tableau1[:xover_debut] + tableau2[xover_debut:xover_fin] + tableau1[xover_fin:]
        enfant2 = tableau2[:xover_debut] + tableau1[xover_debut:xover_fin] + tableau2[xover_fin:]
        return [enfant1, enfant2]

    def _tp_mutate_fn( tableau: 'list[int]', contexte: 'Genetique' ) -> 'list':
        i = random.randint( 0, len( tableau ) - 1 )
        tableau[i] = not tableau[i]
        return tableau

    def _tp_decoder_fn( x: 'list', contexte: 'Genetique' ) -> 'int':
        return Utils.entier( x )

    def _tp_post_fn( x: 'int', contexte: 'Genetique' ) -> 'float':
        return x / contexte.parametres['puissance']
    
    tp_taille               = 100
    tp_expiration           = 5
    tp_bit_nombre           = 8
    tp_mutate_ratio         = 0.01
    tp_min_selection_ratio  = 0.25
    tp_max_selection_ratio  = 0.75
    tp_min_xover_ratio      = 0.25
    tp_max_xover_ratio      = 0.75

    tp_mutate_fn    = _tp_mutate_fn
    tp_fitness_fn   = _tp_fitness_fn
    tp_pre_fn       = _tp_pre_fn
    tp_post_fn      = _tp_post_fn
    tp_encoder_fn   = _tp_encoder_fn
    tp_decoder_fn   = _tp_decoder_fn
    tp_xover_fn     = _tp_xover_fn

    tp_parametres   = {
        'bit_nombre'      : tp_bit_nombre,
        'puissance'       : 1 << tp_bit_nombre,
        'min_xover_ratio' : tp_min_xover_ratio,
        'max_xover_ratio' : tp_max_xover_ratio,
    }
    
    tp_genetique    = Genetique(
        taille      = tp_taille,
        expiration  = tp_expiration,
        parametres  = tp_parametres,
        fitness_fn  = tp_fitness_fn,
        mutate_fn   = tp_mutate_fn,
        pre_fn      = tp_pre_fn,
        post_fn     = tp_post_fn,
        encoder_fn  = tp_encoder_fn,
        decoder_fn  = tp_decoder_fn,
        xover_fn    = tp_xover_fn,
        core        = {
            'crossover': {
                'min_ratio': tp_min_selection_ratio,
                'max_ratio': tp_max_selection_ratio,
            },
            'mutation' : {
                'ratio': tp_mutate_ratio
            }
        }
    )

    tp_iteration    = 6
    tp_garder       = True

    tp_result__     = tp_genetique.run( tp_iteration, tp_garder )

    print( tp_fitness_fn.__doc__ )
    print( 'Nombre de bit d\'encodage  :', tp_bit_nombre )
    print( 'Cycles .................. :', tp_iteration )

    print( '\nTaille Population ....... :', tp_taille )
    print( 'Expiration .............. :', tp_expiration )
    print( '% Selection Crossover ... :', tp_min_selection_ratio, ' - ', tp_max_selection_ratio )
    print( '% Taille Crossover ...... :', tp_min_xover_ratio, ' - ', tp_max_xover_ratio )
    print( '% Mutation .............. :', tp_mutate_ratio )
    print( 'Conserver population .... :', 'Oui' if tp_garder else 'Non' )

    print( '\nx pour f(x) minimum ..... :', tp_result__ )


if __name__ == "__main__":
    start = time.time()
    TP_Projet_IA()
    end = time.time()
    print( 'Execution (ms) .......... :', end - start )
    print('\n\n')
