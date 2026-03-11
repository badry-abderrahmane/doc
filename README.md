# État de l'Art
## Modélisation des émotions de masse lors d'événements psychosociaux à partir des réseaux sociaux : contagion émotionnelle et effets de groupe

**ABDERRAHMANE BADRY**

**FAOUZI MERZOUKI**

---

## 1. Introduction

### Contexte général

L'essor des plateformes de réseaux sociaux a profondément reconfiguré les modalités d'expression collective : lors d'événements psychosociaux majeurs — qu'ils soient sportifs, politiques, sanitaires ou culturels — ces plateformes génèrent en temps réel des volumes massifs de données textuelles, visuelles et multimédias, au sein desquelles des vagues d'émotions — joie, colère, peur, enthousiasme — se propagent de manière non linéaire entre individus et communautés. Comprendre les mécanismes par lesquels ces dynamiques émotionnelles collectives émergent, s'amplifient et influencent les comportements est devenu un enjeu scientifique de premier plan pour la sociologie computationnelle, la gestion des risques sociaux et la communication publique.

### Objectif de l'état de l'art

Ce chapitre vise à examiner l'évolution des techniques d'analyse des émotions, de modélisation de l'influence sociale et de simulation des comportements collectifs, afin de démontrer qu'il existe un verrou scientifique majeur : bien que la classification des émotions individuelles soit désormais avancée grâce aux modèles de Deep Learning, la **modélisation dynamique de la contagion émotionnelle de masse** — dans sa dimension temporelle, multimodale et à grande échelle — lors d'événements psychosociaux spécifiques reste un défi ouvert. Par ailleurs, les enjeux éthiques liés à la collecte et au traitement de données affectives à grande échelle constituent un défi transversal que tout cadre proposé dans ce domaine se doit d'adresser.

### Périmètre de la recherche

Notre étude se concentre sur l'intégration des algorithmes de traitement du langage naturel (NLP), de vision par ordinateur, d'intelligence artificielle générative et de modélisation de graphes dynamiques appliqués aux données sociales en ligne. Nous excluons les approches sociologiques purement qualitatives pour nous focaliser sur la **sociologie computationnelle** et l'exploration de données à grande échelle. Les domaines d'application ciblés comprennent la gestion des événements publics, la prévention des risques sociaux et la communication de crise.

### Structure du chapitre

Ce chapitre présentera d'abord les fondements théoriques de l'analyse des émotions et de la contagion sociale, avant d'analyser thématiquement les approches existantes (NLP unimodal, Deep Learning multimodal, modèles de graphes et systèmes multi-agents). Une synthèse critique mettra ensuite en lumière les limites actuelles, pour finalement introduire le positionnement spécifique de cette thèse.

---

## 2. Fondements théoriques et concepts de base

### 2.1 Définitions clés

**Analyse de sentiment (Sentiment Analysis / Opinion Mining)** : Technique informatique visant à extraire la *valence affective* d'un contenu textuel ou multimodal, c'est-à-dire à déterminer si ce contenu exprime une opinion positive, négative ou neutre envers un sujet donné (Alam et al., 2025 ; Ganesh et al., 2025). Il est important de distinguer cette approche de la reconnaissance des émotions à proprement parler : l'analyse de sentiment opère principalement au niveau de la *polarité*, tandis que la **reconnaissance des émotions (Emotion Recognition)** vise à identifier des états affectifs discrets et psychologiquement fondés, tels que ceux décrits par le modèle d'Ekman (joie, colère, tristesse, peur, surprise, dégoût) ou le modèle de Plutchik (roue des émotions à huit dimensions primaires). Cette distinction est centrale pour la présente thèse : notre ambition dépasse la classification polaire pour cibler la modélisation d'états affectifs complexes et leur dynamique collective, ce que la littérature désigne comme *fine-grained emotion analysis* (Singla & Alhussan, 2024 ; Yan et al., 2025).

**Émotion de masse** : État émotionnel partagé par un groupe ou une communauté, résultant d'une dynamique collective et non d'une simple agrégation d'états individuels. Elle se manifeste par des phénomènes tels que la mobilisation, la polarisation ou la propagation virale d'un affect dominant au sein d'un réseau social lors d'un événement déclencheur.

**Contagion émotionnelle** : Le phénomène par lequel l'état affectif d'un utilisateur est influencé par les émotions exprimées par ses connexions (amis, abonnés) sur une période donnée. Zafarani, Cole et Liu (2010) ont formellement défini ce concept dans le contexte des réseaux sociaux numériques. Soit $\mu$ un utilisateur actif sur un site $s$, $\Lambda_s$ l'ensemble de *tous* les utilisateurs actifs sur ce site (population de référence globale), $U \subset \Lambda_s$ un sous-ensemble d'utilisateurs initiant la propagation, et $m(\mu, t)$ la valeur de sentiment de l'utilisateur $\mu$ au temps $t$. L'humeur moyenne d'un groupe $U$ au temps $t$ est définie par :

$$m(U, t) = \frac{\sum_{\mu \in U} m(\mu, t)}{|U|}$$

On considère qu'une propagation de sentiment a influencé un utilisateur cible $\mu$ entre les instants $t_i$ et $t_j$ ($t_i \leq t_j$) si les deux conditions suivantes sont satisfaites :

$$|m(U, t_i) - m(\mu, t_j)| \leq |m(\Lambda_s, t_i) - m(\mu, t_j)| + b_1 \tag{1}$$

$$|m(U, t_i) - m(\mu, t_j)| \leq |m(U, t_i) - m(\mu, t_i)| + b_2 \tag{2}$$

La condition (1) stipule qu'à l'instant $t_j$, l'humeur de $\mu$ est plus proche de celle du groupe initiateur $U$ que de la population globale $\Lambda_s$ — ce qui indique une influence spécifique du groupe plutôt qu'une dérive aléatoire. La condition (2) stipule que cette proximité s'est accrue dans le temps, c'est-à-dire que $\mu$ s'est rapproché de $U$ entre $t_i$ et $t_j$. Les paramètres $b_1, b_2 \geq 0$ sont des seuils de tolérance (intercepts) qui permettent de modéliser la flexibilité de la définition d'influence : ils peuvent être appris par données ou fixés heuristiquement selon le contexte expérimental (Zafarani et al., 2010). Des études récentes sur la contagion émotionnelle en ligne ont montré que ces effets peuvent persister jusqu'à huit semaines sous forme de cycles récurrents, et que la peur est l'émotion se propageant le plus rapidement, devant la colère et la joie (Tandfonline, 2025).

**Graphe dynamique** : Représentation mathématique $G(t) = (V, E(t))$ d'un réseau social, où l'ensemble des nœuds $V$ représente les utilisateurs et l'ensemble des arêtes $E(t)$ représente leurs interactions au temps $t$ (mentions, retweets, commentaires), évoluant au fil du temps en réponse aux événements et aux comportements observés.

**Effets de groupe** : Phénomènes émergents résultant des interactions entre membres d'une communauté, incluant la **polarisation** (radicalisation progressive vers des positions extrêmes), la **mobilisation** (passage à l'action collective coordonnée) et la **propagation de comportements** (imitation et amplification d'attitudes au sein du réseau).

**Analyse multimodale** : Approche intégrant simultanément plusieurs types de données — texte, image, vidéo, émojis, audio — pour produire une interprétation émotionnelle plus complète et plus précise du contenu publié sur les réseaux sociaux, en exploitant la complémentarité sémantique entre modalités.

### 2.2 Théories fondamentales

Pour modéliser la diffusion de l'influence émotionnelle à travers le réseau, les chercheurs s'appuient sur deux modèles stochastiques classiques issus de la théorie de la diffusion d'information :

- **Le Modèle de Cascade Indépendante (Independent Cascade Model, ICM)** : La diffusion se propage de manière stochastique ; lorsqu'un nœud $u$ est activé (adopte une émotion), il tente d'activer chacun de ses voisins $v$ non encore activés avec une probabilité $p_{u,v}$ fixe. Ce modèle est particulièrement adapté à la modélisation de la propagation virale d'un contenu émotionnel à travers des liens faibles (Kempe, Kleinberg & Tardos, 2003).

- **Le Modèle à Seuil Linéaire (Linear Threshold Model, LTM)** : Un utilisateur adopte une émotion ou un comportement lorsque la proportion pondérée de ses connexions ayant déjà adopté ce comportement dépasse un seuil individuel $\theta_u \in [0,1]$. Ce modèle capture mieux les phénomènes de mobilisation collective par effet de masse critique (Granovetter, 1978).

- **Le Modèle S3EIR** : Chen et al. (2025) proposent une extension du modèle épidémiologique classique SEIR appliquée à la propagation émotionnelle sur les réseaux sociaux. Ce modèle stratifié compartimente la population en états *Susceptible*, *Exposed* (exposé mais non encore émotionnellement activé), *Emotionally Infected* (contagieux), et *Recovered* (revenu à un état de base). Sa force réside dans sa capacité à modéliser simultanément plusieurs émotions en compétition au sein du même réseau, ainsi que les effets de rétroaction entre émotions positives et négatives — une avancée significative par rapport aux modèles ICM et LTM à émotion unique (*Frontiers in Communication*, DOI: 10.3389/fcomm.2025.1582974).

À ces modèles s'ajoutent les théories d'**Opinion Dynamics** issues de la physique sociale (modèles de Deffuant, Hegselmann-Krause), qui modélisent l'évolution continue des opinions dans un réseau d'agents en interaction, et les théories de la **contagion sociale** validées empiriquement — notamment l'étude longitudinale de Fowler & Christakis (2008) sur la propagation du bonheur dans un réseau social réel sur 20 ans, démontrant l'existence de clusters émotionnels stables dans un graphe de relations interpersonnelles.

### 2.3 Historique rapide

L'étude de l'influence sociale s'appuyait historiquement sur des modèles sociologiques statiques (matrices sociométriques, modèles d'opinion) incapables de traiter de grands volumes de données en temps réel (Katz & Lazarsfeld, 1955). Sur le plan computationnel, les premières approches reposaient sur des **lexiques de sentiment** (AFINN, VADER, SentiWordNet), simples mais limités face à l'ambiguïté linguistique. Avec l'avènement du Big Data, le domaine a évolué vers des modèles statistiques de **Machine Learning** (SVM, Naive Bayes, Random Forest), puis vers les **réseaux de neurones profonds** (LSTM, CNN). L'apparition des architectures **Transformer** (BERT, RoBERTa, XLM-R) a constitué une rupture méthodologique majeure, portant la précision de la classification de sentiment à des niveaux inédits (Karim et al., 2025 ; Alam et al., 2025). Aujourd'hui, les **Grands Modèles de Langage (LLM)** comme GPT-3/4 ouvrent de nouvelles perspectives pour l'analyse fine des nuances affectives (Singla & Alhussan, 2024), tandis que les architectures hybrides Transformer-GNN représentent la frontière technologique de 2025 (Yan et al., 2025 ; COLING, 2025).

La modélisation de la **contagion émotionnelle** dans les réseaux numériques est un domaine plus récent. Les premières études empiriques, comme celle de Zafarani et al. (2010) sur LiveJournal, ont posé les bases formelles du problème. Les travaux ultérieurs ont progressivement intégré des représentations graphiques, puis des approches multimodales, et les développements les plus récents (2025) tendent vers des modèles hybrides intégrant simultanément la dynamique temporelle, la multimodalité et la structure du réseau — bien que la combinaison complète de tous ces aspects dans un cadre unifié reste à construire.

---

## 3. Analyse thématique de la littérature

### 3.1 Taxonomie des solutions existantes

Les recherches actuelles peuvent être regroupées en cinq grandes catégories technologiques : (1) approches lexicales et à base de règles, (2) Machine Learning classique, (3) Deep Learning et architectures Transformer, (4) analyse multimodale (texte + image + vidéo), et (5) modèles structurels basés sur les graphes et systèmes multi-agents. Ces catégories ne sont pas mutuellement exclusives ; les travaux les plus récents tendent vers leur intégration progressive, notamment autour d'architectures hybrides Transformer-GNN.

### 3.2 Analyse des méthodes dominantes

#### a) Approches lexicales et Machine Learning classique

Les premières méthodes computationnelles s'appuyaient sur des dictionnaires de sentiment (AFINN, VADER, SentiWordNet) qui assignent un score de polarité à chaque mot (Zafarani et al., 2010). Bien qu'interprétables et rapides, ces approches échouent systématiquement face au sarcasme, à la négation et aux expressions idiomatiques (Ganesh et al., 2025 ; Alam et al., 2025).

Les algorithmes de Machine Learning classique — **Machines à Vecteurs de Support (SVM)**, Naive Bayes et Régression Logistique — combinés à des représentations vectorielles telles que TF-IDF, ont amélioré significativement les performances (Ganesh et al., 2025). L'étude de Karim et al. (2025) sur le dataset Sentiment140 rapporte des performances de SVM comparables à celles de Naive Bayes sur les données textuelles courtes, mais souligne leur dépendance à l'ingénierie manuelle des caractéristiques. L'analyse de sentiments politiques avec une SVM linéaire a atteint 91,18 % de précision (Singla & Alhussan, 2024), illustrant leur robustesse dans des contextes thématiquement bien délimités. Ces modèles peinent néanmoins face à l'**ambiguïté contextuelle**, au **code-switching** (mélange de langues) et à l'**évolution temporelle du langage** (néologismes, argot, émojis), des phénomènes particulièrement fréquents dans le contexte des événements psychosociaux.

#### b) Deep Learning et architectures Transformer

L'utilisation des **Réseaux de Mémoire à Court et Long Terme (LSTM)** et des **Réseaux de Neurones Convolutifs (CNN)** a permis de capturer les dépendances séquentielles et les structures locales dans les textes (Alam et al., 2025 ; Ganesh et al., 2025). Ces architectures surpassent les méthodes classiques sur les tâches de classification d'émotions fines, notamment en capturant les relations entre mots distants dans une phrase.

Une rupture méthodologique majeure a été opérée par les architectures **Transformer** et les modèles pré-entraînés. **BERT** (Bidirectional Encoder Representations from Transformers) capture les relations sémantiques bidirectionnelles grâce à son mécanisme d'attention multi-tête, atteignant une précision de **83,48 %** (Précision : 79,37 %, Rappel : 90,60 %, F1 : 84,61 %) dans la classification de sentiment sur Twitter (Karim et al., 2025). Ses variantes multilingues (**XLM-R**, **mBERT**) permettent d'étendre l'analyse au-delà des corpus anglophones, adressant partiellement le problème de la généralisation multilingue.

Une avancée récente et significative est représentée par le réseau **Emotion-RGC Net** proposé par Yan et al. (2025). Cette architecture hybride combine **RoBERTa** pour l'encodage sémantique profond, un **GNN (Graph Neural Network)** pour modéliser les relations contextuelles inter-phrases, et un **CRF (Conditional Random Field)** pour le décodage séquentiel structuré des étiquettes émotionnelles. Cette combinaison permet de traiter l'émotion non pas comme un phénomène isolé au niveau du token, mais comme un phénomène structurellement dépendant du contexte conversationnel et des relations entre utilisateurs. L'architecture atteint **89,70 % de précision** dans la reconnaissance des émotions sur des données de réseaux sociaux, surpassant les approches uniquement basées sur les Transformer, et constitue un argument empirique fort en faveur de l'intégration de la structure du réseau dans les modèles d'analyse émotionnelle (*PLOS ONE*, DOI: 10.1371/journal.pone.0318524).

Les **Grands Modèles de Langage (LLM)** — GPT-3, GPT-4, LLaMA — constituent l'avancée la plus récente (Singla & Alhussan, 2024). Leur déploiement dans un cadre scientifique rigoureux soulève cependant plusieurs réserves importantes. D'une part, ces modèles sont sujets à des **hallucinations** et à un **biais d'alignement** introduit par le fine-tuning par renforcement à partir de retours humains (RLHF), ce qui peut induire des classifications affectives superficielles ou culturellement biaisées. D'autre part, ils peinent sur les **implicatures culturelles spécifiques** (sous-entendus liés à un contexte géographique, linguistique ou événementiel particulier) qui sont précisément au cœur de l'analyse des émotions lors d'événements psychosociaux. Leur utilisation fiable dans ce domaine requiert donc un **fine-tuning sur des corpus annotés spécialisés** et/ou une approche de type **RAG (Retrieval-Augmented Generation)** pour ancrer les inférences dans des données contextuellement vérifiées. À ces limites s'ajoute leur coût computationnel élevé, qui constitue un obstacle à leur déploiement à grande échelle en temps réel.

#### c) Analyse multimodale : texte, image et vidéo

Les interactions sur les réseaux sociaux ne sont plus exclusivement textuelles. Les contenus publiés lors d'événements psychosociaux (mèmes, images de célébration ou de protestation, vidéos virales) portent une charge émotionnelle que le texte seul ne peut capturer. L'analyse multimodale s'impose donc comme une nécessité pour rendre compte de la complexité sémiotique des échanges en ligne.

Sur le plan de la **vision par ordinateur**, les architectures **Vision Transformer (ViT)** (Dosovitskiy, 2020) permettent d'analyser le contenu visuel des images en utilisant le même mécanisme d'attention que BERT, atteignant des performances comparables aux CNN pour l'extraction de la valence affective d'une image (expressions faciales, composition chromatique, symbolique visuelle). Pour l'analyse vidéo, **VideoBERT** (Sun et al., 2019) étend l'approche Transformer aux séquences temporelles, permettant une analyse conjointe des trames visuelles et de la bande audio. Les architectures de **fusion multimodale** — fusion précoce (*early fusion*), tardive (*late fusion*) ou hybride — combinent ensuite les représentations issues de chaque modalité pour produire une prédiction émotionnelle unifiée, permettant notamment de résoudre des ambiguïtés telles que le sarcasme exprimé par la conjonction d'un texte positif et d'une image ironique. L'analyse de Karim et al. (2025) souligne que les modèles émergents comme **ELMo** et **GPT-3** améliorent le traitement des textes courts et des contenus multimodaux dynamiques.

#### d) Modèles de Graphes et Systèmes Multi-Agents

Tandis que les modèles d'analyse de contenu — NLP et Vision — excellent à caractériser l'émotion émise par un nœud isolé, ils demeurent structurellement aveugles à la topologie du réseau et aux mécanismes de propagation inter-utilisateurs. C'est pourquoi le recours à la théorie des graphes et aux modèles de diffusion est indispensable pour modéliser la dimension collective et relationnelle de la contagion émotionnelle.

Les **Réseaux de Neurones sur Graphes (GNN)** permettent d'apprendre des représentations vectorielles des nœuds en tenant compte de leur contexte local et global dans le réseau, via des mécanismes d'agrégation de voisinage. Appliqués à la détection d'influence, ils identifient les nœuds d'influence majeurs en calculant les degrés de centralité (centralité de degré, d'intermédiarité, de proximité). Ethan & Smith (2023) proposent un cadre combinant GNN, analyse prédictive et analyse comportementale pour détecter les schémas d'influence sociale en temps réel sur des données multi-plateformes. Plus récemment, des architectures combinant GNN et **équations différentielles ordinaires neurales (Neural ODE)** — désignées *Dynamic Graph Neural ODE Networks* — permettent de modéliser en continu l'évolution des représentations de nœuds entre les instants d'interaction, offrant ainsi une modélisation temporelle plus fine et plus réaliste que les approches discrètes. Appliquées à la reconnaissance d'émotions multimodales, ces architectures parviennent à capturer les dynamiques émotionnelles inter-modales à travers le temps (COLING, 2025), représentant une avancée vers les graphes temporels continus.

Les algorithmes de **détection de communautés**, tels que la **méthode de Louvain**, identifient des groupes d'utilisateurs partageant des comportements émotionnels similaires au sein du réseau (Ethan & Smith, 2023). Ces communautés constituent les unités naturelles pour l'étude des effets de groupe. Les **modèles de diffusion ICM et LTM** permettent de simuler la propagation des émotions à travers le graphe et de prédire l'ampleur d'un phénomène de contagion (Kempe, Kleinberg & Tardos, 2003 ; Ethan & Smith, 2023). Le **modèle S3EIR** de Chen et al. (2025) offre quant à lui une perspective épidémiologique stratifiée, en distinguant les phases d'exposition latente, d'infection émotionnelle active et de récupération affective au sein du graphe — permettant une modélisation plus fidèle des phénomènes de saturation émotionnelle observés lors d'événements prolongés. L'étude fondatrice de Zafarani et al. (2010) sur LiveJournal (16 444 bloggers, 131 846 liens, 475 932 publications) a démontré empiriquement que les utilisateurs ayant plus d'amis et moins prolifiques présentent une plus grande susceptibilité à la propagation émotionnelle, et que cette propagation s'opère dans des fenêtres temporelles inférieures à quatre mois.

Sur le plan empirique, des études récentes portant sur la contagion émotionnelle lors de mouvements de protestation ont mis en évidence des mécanismes de diffusion spécifiques aux événements collectifs à fort enjeu politique : la mobilisation émotionnelle y suit une courbe non linéaire marquée par des cycles de propagation et de récupération, les émotions de peur et de colère circulant par des canaux distincts de la joie ou de l'espoir (TPM, 2025). Ces observations s'inscrivent directement dans la problématique des événements psychosociaux ciblés dans cette thèse.

Les **Systèmes Multi-Agents (SMA)** complètent ce dispositif en permettant de simuler le comportement collectif émergent à partir de règles d'interaction individuelles, offrant un cadre de test pour des scénarios de polarisation ou de mobilisation.

#### e) Traitement en Temps Réel

L'étude des émotions lors d'événements psychosociaux exige une capacité d'analyse en flux continu, les dynamiques collectives pouvant évoluer en quelques minutes. L'intégration de frameworks de streaming tels qu'**Apache Kafka** et **Apache Flink** avec des modèles prédictifs permet d'analyser la propagation de l'influence en direct, dépassant les approches batch sur des datasets statiques comme Sentiment140 (Karim et al., 2025). Ces architectures distribuées permettent de traiter des millions d'interactions par seconde et de détecter les pics émotionnels au moment où ils se produisent (Ethan & Smith, 2023).

### 3.3 Étude des technologies émergentes

**IA Générative et LLM** : Les LLM de dernière génération (GPT-4, LLaMA 3, Mistral) ne se limitent plus à la classification : ils peuvent générer des synthèses émotionnelles contextualisées et produire des explications en langage naturel sur la dynamique affective observée (Singla & Alhussan, 2024). Leur capacité à traiter de longues séquences contextuelles les rend particulièrement adaptés à l'analyse de fils de discussion lors d'événements. Leur usage doit néanmoins être encadré par des méthodes de validation et d'ancrage contextuel (RAG, fine-tuning supervisé) pour en garantir la fiabilité scientifique. Des travaux récents explorent explicitement l'utilisation des LLM pour la **mesure automatisée de la polarisation** sur les réseaux sociaux, démontrant leur capacité à quantifier les dynamiques de division sociale sur des corpus de grande taille — ouvrant la voie à une nouvelle catégorie d'applications analytiques dans le domaine des émotions politiques collectives (Springer, 2025).

**Explainable AI (XAI)** : Face aux préoccupations éthiques liées à la dimension "boîte noire" des modèles de Deep Learning, les approches d'IA explicable — LIME, SHAP, visualisation des mécanismes d'attention — permettent de rendre interprétables les décisions des modèles (Alam et al., 2025 ; Singla & Alhussan, 2024). Dans le contexte de l'analyse des comportements politiques ou sociaux, l'explicabilité n'est pas seulement une vertu technique : elle constitue une condition nécessaire à la légitimité scientifique et éthique du système.

**Graphes dynamiques temporels** : Des architectures combinant GNN et mécanismes d'attention temporelle (Temporal Graph Networks, TGN) permettent de capturer l'évolution des structures de réseau au cours d'un événement. Une extension récente de cette famille de modèles — les **Dynamic Graph Neural ODE Networks** (COLING, 2025) — modélise en continu l'évolution des embeddings de nœuds via des équations différentielles, permettant une caractérisation plus précise des trajectoires émotionnelles lors de séquences d'événements rapprochés, et représentant l'état de l'art actuel pour la reconnaissance multimodale d'émotions dans les graphes temporels.

**Ressources multilingues et benchmarks de polarisation** : La disponibilité de ressources annotées à large spectre linguistique constitue un levier majeur pour l'évaluation des modèles dans des contextes culturels diversifiés. Le benchmark **POLAR** récemment publié (Naseem et al., 2025) rassemble 110 000 instances dans **22 langues**, constituant à ce jour la ressource multilingue la plus complète pour l'évaluation de la polarisation émotionnelle sur les réseaux sociaux. Cette ressource ouvre des perspectives concrètes pour la généralisation des modèles au-delà de l'anglais, en réponse partielle aux lacunes multilingues identifiées dans la littérature (arXiv, 2505.20624).

---

## 4. Analyse critique et synthèse

### 4.1 Tableau comparatif des approches

| Approche | Avantages | Inconvénients | Pertinence pour la thèse |
|---|---|---|---|
| Lexicale / Règles | Simple, rapide, interprétable | Échoue sur le sarcasme, la négation, l'argot | Faible |
| ML classique (SVM, RF) | Efficace sur corpus étiquetés, bon rapport coût/performance | Ingénierie manuelle des features ; pas de dynamisme temporel | Moyenne |
| Deep Learning (LSTM, CNN) | Capture les dépendances séquentielles et spatiales | Mono-modal ; limité sur les textes courts | Moyenne |
| Transformers (BERT, XLM-R) | Haute précision, contexte profond, multilingue | Coût de calcul élevé ; dynamisme temporel limité | **Haute** |
| Transformer-GNN hybride (RGC Net) | Intègre encodage sémantique + structure du réseau | Nécessite des données de graphe annoté | **Très haute** |
| LLM / IA Générative | Analyse nuancée, génération de synthèses contextuelles | Hallucinations, biais RLHF, coût élevé ; nécessite RAG/fine-tuning | **Haute** |
| Multimodal (ViT, VideoBERT) | Intègre texte + image + vidéo | Complexité de fusion ; rareté des datasets annotés | **Très haute** |
| Graphes / GNN | Modélisation structurelle, propagation, détection d'influence | Scalabilité sur graphes dynamiques massifs | **Très haute** |
| Graph Neural ODE (temporel continu) | Dynamique temporelle continue entre événements | Complexité mathématique ; coût d'entraînement élevé | **Très haute** |
| Modèles ICM / LTM / S3EIR | Simulation de propagation, scénarios prédictifs, multi-émotions | Paramétrage difficile ; hypothèses simplificatrices (ICM/LTM) | **Haute** |
| Systèmes Multi-Agents | Comportements émergents, simulation d'événements | Validation empirique difficile à grande échelle | **Haute** |
| Temps réel (Kafka, Flink) | Analyse in-stream, réactivité aux événements | Arbitrage latence / précision ; infrastructure complexe | **Haute** |

### 4.2 Forces et faiblesses globales de la littérature

Sur le plan des forces, la revue systématique d'Alam et al. (2025), couvrant 91 articles publiés entre 2010 et 2024, confirme que les modèles basés sur les architectures Transformer constituent l'état de l'art pour la classification de sentiment sur les données de réseaux sociaux, avec une amélioration continue de la précision et de la généralisation multilingue. Par ailleurs, les modèles de graphes se révèlent hautement efficaces pour simuler la propagation des émotions et identifier les nœuds d'influence ; l'étude d'Ethan & Smith (2023) démontre la viabilité d'un cadre combinant NLP, GNN et analyse prédictive pour la détection d'influence sociale en temps réel. Les travaux de 2025 montrent une convergence significative vers des architectures hybrides Transformer-GNN (Yan et al., 2025) et des modèles temporels continus (COLING, 2025), signalant une accélération de la maturité technologique dans le domaine.

La littérature présente néanmoins plusieurs limites persistantes. La **détection du sarcasme, de l'ironie et de l'ambiguïté** linguistique demeure un défi technique non résolu, particulièrement fréquent lors d'événements politiques ou sportifs. Le **biais anglophone** des datasets et des modèles pré-entraînés (Sentiment140, SemEval) limite leur applicabilité dans des contextes multilingues ou dialectaux, bien que le benchmark POLAR (Naseem et al., 2025) représente un progrès notable sur ce point. Le **déséquilibre des corpus** entraîne une sur-représentation de certaines émotions primaires (joie, colère) au détriment des émotions secondaires ou ambivalentes. Le **manque d'explicabilité** des modèles profonds soulève des préoccupations éthiques légitimes (Singla & Alhussan, 2024 ; Alam et al., 2025) lorsque ces outils sont utilisés pour analyser des comportements politiques ou sociaux. Enfin, les **préoccupations relatives à la vie privée et au consentement éclairé** lors de la collecte et de l'analyse de données personnelles à grande échelle constituent un impératif éthique que la littérature aborde encore de manière insuffisante (Alam et al., 2025).

### 4.3 Identification des verrous scientifiques

L'analyse de la littérature révèle cinq lacunes majeures, directement liées à la problématique de cette thèse :

**Gap 1 — Le manque de dynamisme temporel lié aux événements :** La grande majorité des études exploite des datasets statiques (Sentiment140, IMDb) (Karim et al., 2025). L'analyse de la contagion émotionnelle *au cours* d'un événement psychosocial spécifique — avec ses phases d'anticipation, de pic émotionnel et de retombée — est quasi inexistante dans la littérature. Les modèles existants ne capturent pas la **trajectoire temporelle** de l'émotion de masse, une limite que Zafarani et al. (2010) reconnaissent explicitement comme direction prioritaire de recherche future. Ce gap est d'autant plus pressant que des études récentes ont révélé que les effets de contagion émotionnelle en ligne peuvent persister jusqu'à huit semaines sous forme de cycles récurrents, et que des émotions différentes — notamment la peur — se propagent sur des horizons temporels distincts (Tandfonline, 2025) : autant de dynamiques qu'un modèle statique est structurellement incapable de capturer.

**Gap 2 — L'isolement des modalités et de la structure du réseau :** Peu de frameworks parviennent à analyser simultanément le contenu multimodal (textes, images, vidéos publiés par les utilisateurs) et la dynamique structurelle du réseau (mécanismes de propagation et de polarisation). Ces deux dimensions sont généralement traitées séparément (Alam et al., 2025 ; Ethan & Smith, 2023), et Ethan & Smith (2023) soulignent explicitement que leur combinaison dans un cadre unifié reste une direction de recherche ouverte. Les Dynamic Graph Neural ODE Networks (COLING, 2025) constituent une avancée prometteuse sur ce point, mais leur validation sur des événements psychosociaux réels à grande échelle reste à réaliser.

**Gap 3 — L'absence de modélisation des effets de groupe spécifiques aux événements :** Les phénomènes de **polarisation**, de **mobilisation collective** et de **propagation comportementale** lors d'événements psychosociaux identifiés sont insuffisamment modélisés. La question de savoir *comment* et *dans quelles conditions* une émotion individuelle devient une émotion de masse lors d'un événement spécifique ne dispose pas de réponse formalisée et empiriquement validée à ce jour (Singla & Alhussan, 2024 ; Ganesh et al., 2025). Si le benchmark POLAR (Naseem et al., 2025) offre désormais une ressource d'évaluation multilingue pour la polarisation, il demeure axé sur la détection statique et ne propose pas de modélisation dynamique des mécanismes collectifs d'escalade ou de polarisation au cours d'un événement.

**Gap 4 — Le manque d'applications prédictives opérationnelles :** Les travaux existants privilégient la classification et la détection rétrospective (Alam et al., 2025). Les modèles permettant de **prédire** l'évolution émotionnelle d'une communauté lors d'un événement en cours — et d'anticiper un pic de polarisation ou l'émergence d'une crise — restent à développer. Ethan & Smith (2023) proposent une ébauche de framework prédictif pour la détection d'influence, mais celui-ci ne s'applique pas à la contagion émotionnelle de masse dans le contexte d'événements spécifiques. Le modèle S3EIR (Chen et al., 2025) offre un cadre de simulation prédictif multi-émotions, mais il n'intègre pas encore les dimensions multimodales ni la granularité fine des états affectifs.

**Gap 5 — L'insuffisance des garanties éthiques et de l'explicabilité :** Les travaux de la littérature soulèvent les risques liés au manque d'explicabilité des modèles et aux enjeux de vie privée (Alam et al., 2025 ; Singla & Alhussan, 2024), mais peu de frameworks proposent des solutions intégrées. L'absence de mécanismes d'**anonymisation**, de **consentement éclairé** et de **validation explicable** constitue un obstacle à la fois éthique et à la confiance scientifique dans les résultats produits, particulièrement dans des contextes aussi sensibles que l'analyse de comportements politiques ou communautaires.

---

## 5. Positionnement de la thèse

### 5.1 L'opportunité de recherche

La convergence récente de plusieurs maturités technologiques crée les conditions favorables à une contribution scientifique novatrice. D'une part, les **Grands Modèles de Langage** et les architectures multimodales permettent désormais de traiter les nuances affectives complexes à une échelle auparavant inaccessible. D'autre part, les outils de **graphes dynamiques** et de **simulation agentique** atteignent un niveau de maturité permettant leur application à des réseaux sociaux réels de millions d'utilisateurs. L'émergence en 2025 d'architectures hybrides Transformer-GNN (Yan et al., 2025), de modèles de diffusion multi-émotions (Chen et al., 2025), de benchmarks multilingues à grande échelle (Naseem et al., 2025) et de graphes temporels continus (COLING, 2025) confirme que le domaine entre dans une phase de synthèse technologique qui rend possible — et nécessaire — la conception d'un cadre intégré. L'abondance de données produites lors d'événements psychosociaux majeurs fournit enfin un terrain d'expérimentation empirique inédit. Ces conditions réunies offrent l'opportunité de **contribuer à l'élaboration d'un cadre holistique novateur** de modélisation des émotions de masse lors d'événements psychosociaux, en réponse aux cinq verrous scientifiques identifiés ci-dessus.

### 5.2 Hypothèses de travail

Nous posons l'hypothèse qu'un couplage méthodologique entre :
1. Une **architecture d'analyse multimodale** (traitant conjointement textes, images et vidéos) basée sur des modèles Transformer de type BERT/ViT fine-tunés sur des corpus affectifs spécialisés,
2. Un **graphe dynamique temporel** modélisant l'évolution des interactions sociales au cours de l'événement via des Temporal Graph Networks ou des Graph Neural ODE,
3. Des **simulations agentiques** calibrées sur les données empiriques du réseau, intégrant des modèles de diffusion de type S3EIR ou ICM/LTM appris par données,

permettra de pallier les lacunes actuelles et de modéliser avec une plus grande fidélité les **pics de polarisation**, les **vagues de mobilisation** et les **trajectoires de contagion émotionnelle** au sein d'une communauté lors d'un événement psychosocial, tout en garantissant l'explicabilité et le respect de la vie privée des utilisateurs.

### 5.3 Objectifs spécifiques

Par rapport aux recherches existantes, cette thèse vise à :

1. **Concevoir un pipeline de classification émotionnelle multimodale** capable d'interpréter conjointement le texte, les images et les vidéos publiés lors d'un événement psychosocial ciblé, en dépassant la simple polarité positive/négative pour atteindre une granularité d'états affectifs discrets (modèle d'Ekman étendu), en réponse au Gap 2.

2. **Développer un graphe dynamique temporel des interactions** pour modéliser l'évolution de la structure du réseau et le seuil de perméabilité émotionnelle des utilisateurs au cours du temps, en intégrant les modèles de diffusion (ICM, LTM, S3EIR) dans un cadre appris par données, en réponse aux Gaps 1 et 2.

3. **Analyser et quantifier les effets de groupe** (polarisation, mobilisation, propagation comportementale) par des méthodes de détection de communautés et d'analyse de centralité adaptées aux graphes dynamiques, en réponse au Gap 3.

4. **Créer des modèles de simulation prédictive** — combinant Deep Learning et systèmes multi-agents — permettant d'anticiper la trajectoire de l'émotion de masse lors d'un événement en cours et de prévenir l'émergence de crises ou de phénomènes de radicalisation émotionnelle, en réponse au Gap 4.

5. **Valider le cadre sur des événements psychosociaux réels** (événements sportifs, élections, crises sanitaires) en construisant des datasets multimodaux annotés et en publiant les protocoles d'évaluation, contribuant ainsi à la reproductibilité scientifique, avec un protocole d'évaluation multilingue tirant parti des ressources comme POLAR (Naseem et al., 2025).

6. **Intégrer des garanties éthiques et d'explicabilité** au sein du pipeline : mise en place de mécanismes d'anonymisation des données collectées, utilisation de méthodes XAI (SHAP, attention visualization) pour valider et interpréter les décisions du modèle, et conformité aux principes FAIR (*Findable, Accessible, Interoperable, Reusable*) pour la gestion des données, en réponse au Gap 5.

---

## Références bibliographiques

- Alam, M. S., Mrida, M. S. H., & Rahman, M. A. (2025). Sentiment analysis in social media: How data science impacts public opinion knowledge integrates NLP with AI. *American Journal of Scholarly Research and Innovation*, 4(1), 63–100.
- Chen, X., et al. (2025). A stratified epidemic model (S3EIR) for emotional contagion propagation on social networks. *Frontiers in Communication*. DOI: 10.3389/fcomm.2025.1582974.
- Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2020). An image is worth 16x16 words: Transformers for image recognition at scale. *arXiv*, 2010.11929.
- Ethan, A., & Smith, J. (2023). AI-Powered Big Data Analytics for Social Influence Detection. *International Journal of Machine Learning Research in Cybersecurity and Artificial Intelligence*, 14(1).
- Fowler, J., & Christakis, N. (2008). Dynamic spread of happiness in a large social network: longitudinal analysis over 20 years in the Framingham Heart Study. *British Medical Journal*, 337, a2338.
- Ganesh, K., Pallavi, Y. S., Swapna, M., Srinivas, P., & Saikumar, B. (2025). AI-Powered Sentiment Analysis of Social Media: Trends, Challenges, and Insights. *International Journal of Science, Engineering and Technology*, 13(2).
- Granovetter, M. (1978). Threshold models of collective behavior. *American Journal of Sociology*, 83(6), 1420–1443.
- Karim, S. M. R. U., Rasul, R. A., & Sultana, T. (2025). Sentiment Analysis of Social Media Data for Predicting Consumer Behavior Trends Using Machine Learning. *arXiv*, 2510.19656.
- Katz, E., & Lazarsfeld, P. F. (1955). *Personal Influence: The Part Played by People in the Flow of Mass Communications*. Free Press.
- Kempe, D., Kleinberg, J., & Tardos, E. (2003). Maximizing the spread of influence through a social network. *Proceedings of the 9th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 137–146.
- Naseem, U., et al. (2025). POLAR: A multilingual benchmark for polarization analysis in social media. *arXiv*, 2505.20624.
- Singla, M. K., & Alhussan, A. A. (2024). A Review of Artificial Intelligence for Sentiment Analysis in Social Media Data. *Metaheuristic Optimization Review*, 2(2), 1–13.
- Sun, C., Myers, A., Vondrick, C., Murphy, K., & Schmid, C. (2019). VideoBERT: A joint model for video and language representation learning. *Proceedings of ICCV 2019*, 7464–7473.
- Yan, H., et al. (2025). Emotion-RGC Net: A hybrid RoBERTa-GNN-CRF architecture for fine-grained emotion recognition in social media. *PLOS ONE*. DOI: 10.1371/journal.pone.0318524.
- Zafarani, R., Cole, W. D., & Liu, H. (2010). Sentiment Propagation in Social Networks: A Case Study in LiveJournal. *SBP 2010, LNCS 6007*, 413–420.
