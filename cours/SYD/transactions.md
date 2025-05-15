# Gestion des transactions dans les bases de données

## Propriétés ACID

Les propriétés ACID sont un ensemble de caractéristiques qui garantissent qu'une transaction de base de données est traitée de manière fiable. ACID est un acronyme qui représente Atomicité, Cohérence, Isolation et Durabilité.

### Atomicité
L'atomicité garantit que chaque transaction est traitée comme une unité indivisible, qui réussit complètement ou échoue complètement. Si une partie d'une transaction échoue, l'ensemble de la transaction échoue et l'état de la base de données reste inchangé.

Mécanismes d'implémentation :
- Journalisation (logging)
- Shadow paging
- Utilisation de verrous (locks)

### Cohérence
La cohérence assure que toute transaction amène la base de données d'un état valide à un autre état valide. Les contraintes d'intégrité, cascades, déclencheurs et toute combinaison de règles définies doivent être respectées.

Mécanismes pour maintenir la cohérence :
- Contraintes déclaratives (clés primaires, clés étrangères)
- Déclencheurs (triggers)
- Procédures stockées validant les données

### Isolation
L'isolation garantit que l'exécution simultanée de transactions produit le même état que si les transactions avaient été exécutées séquentiellement. Elle empêche qu'une transaction en cours ne soit affectée par d'autres.

Niveaux d'isolation standard :
1. **Read Uncommitted** : Permet les lectures sales (dirty reads)
2. **Read Committed** : Empêche les lectures sales mais autorise les lectures non reproductibles
3. **Repeatable Read** : Empêche les lectures non reproductibles mais autorise les lectures fantômes
4. **Serializable** : Le niveau le plus élevé d'isolation, empêche tous les problèmes d'isolation

### Durabilité
La durabilité garantit que, une fois qu'une transaction a été validée, ses effets persistent même en cas de panne système.

Mécanismes assurant la durabilité :
- Journaux transactionnels (transaction logs)
- Stockage non volatile
- Sauvegardes et réplication

## Contrôle de concurrence

Le contrôle de concurrence est essentiel pour les systèmes multi-utilisateurs où plusieurs transactions peuvent accéder simultanément aux mêmes données.

### Verrouillage (Locking)
Le verrouillage est une technique qui restreint l'accès aux ressources pendant que les transactions les utilisent.

Types de verrous :
- **Verrous partagés (Shared locks)** : Permettent à plusieurs transactions de lire une ressource
- **Verrous exclusifs (Exclusive locks)** : Empêchent toute autre transaction d'accéder à la ressource

### Contrôle de concurrence multi-version (MVCC)
Le MVCC maintient plusieurs versions des données pour permettre aux transactions d'accéder à une version cohérente des données sans bloquer les autres transactions.

Avantages du MVCC :
- Meilleure scalabilité dans les environnements fortement concurrents
- Réduction des contentions
- Support des requêtes de longue durée sans bloquer les mises à jour

### Problèmes de concurrence
1. **Lecture sale (Dirty read)** : Une transaction lit des données qui n'ont pas encore été validées
2. **Lecture non reproductible (Non-repeatable read)** : Une transaction relit des données et constate des modifications
3. **Lecture fantôme (Phantom read)** : Une transaction exécute à nouveau une requête et trouve de nouveaux enregistrements

## Récupération après panne

La récupération après panne est la capacité d'un SGBD à restaurer la base de données dans un état cohérent après une défaillance matérielle ou logicielle.

### Journalisation Write-Ahead (WAL)
Le WAL garantit que les modifications sont d'abord écrites dans un journal avant d'être appliquées à la base de données, ce qui permet de rejouer ou d'annuler des transactions en cas de panne.

### Stratégies de récupération
1. **UNDO** : Annulation des transactions non validées
2. **REDO** : Réapplication des transactions validées dont les modifications n'ont pas été écrites sur disque
3. **UNDO/REDO** : Combinaison des deux approches pour une récupération complète 