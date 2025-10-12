from __future__ import annotations

from river import base

from .hoeffding_tree import HoeffdingTree
from .nodes.branch import DTBranch
from .nodes.htc_nodes import LeafMajorityClass, LeafNaiveBayes, LeafNaiveBayesAdaptive
from .nodes.leaf import HTLeaf
from .split_criterion import GiniSplitCriterion, HellingerDistanceCriterion, InfoGainSplitCriterion
from .splitter import GaussianSplitter, Splitter


class HoeffdingTreeClassifier(HoeffdingTree, base.Classifier):
    """Hoeffding Tree or Very Fast Decision Tree classifier.

    Parameters
    ----------
    grace_period
        Number of instances a leaf should observe between split attempts.
    max_depth
        The maximum depth a tree can reach. If `None`, the tree will grow until
          the system recursion limit.
    split_criterion
        Split criterion to use.</br>
        - 'gini' - Gini</br>
        - 'info_gain' - Information Gain</br>
        - 'hellinger' - Helinger Distance</br>
    delta
        Significance level to calculate the Hoeffding bound. The significance level is given by
        `1 - delta`. Values closer to zero imply longer split decision delays.
    tau
        Threshold below which a split will be forced to break ties.
    leaf_prediction
        Prediction mechanism used at leafs.</br>
        - 'mc' - Majority Class</br>
        - 'nb' - Naive Bayes</br>
        - 'nba' - Naive Bayes Adaptive</br>
    nb_threshold
        Number of instances a leaf should observe before allowing Naive Bayes.
    nominal_attributes
        List of Nominal attributes identifiers. If empty, then assume that all numeric
        attributes should be treated as continuous.
    splitter
        The Splitter or Attribute Observer (AO) used to monitor the class statistics of numeric
        features and perform splits. Splitters are available in the `tree.splitter` module.
        Different splitters are available for classification and regression tasks. Classification
        and regression splitters can be distinguished by their property `is_target_class`.
        This is an advanced option. Special care must be taken when choosing different splitters.
        By default, `tree.splitter.GaussianSplitter` is used if `splitter` is `None`.
    binary_split
        If True, only allow binary splits.
    min_branch_fraction
        The minimum percentage of observed data required for branches resulting from split
        candidates. To validate a split candidate, at least two resulting branches must have
        a percentage of samples greater than `min_branch_fraction`. This criterion prevents
        unnecessary splits when the majority of instances are concentrated in a single branch.
    max_share_to_split
        Only perform a split in a leaf if the proportion of elements in the majority class is
        smaller than this parameter value. This parameter avoids performing splits when most
        of the data belongs to a single class.
    max_size
        The max size of the tree, in mebibytes (MiB).
    memory_estimate_period
        Interval (number of processed instances) between memory consumption checks.
    stop_mem_management
        If True, stop growing as soon as memory limit is hit.
    remove_poor_attrs
        If True, disable poor attributes to reduce memory usage.
    merit_preprune
        If True, enable merit-based tree pre-pruning.
    split_callback
        Optional callback function that will be invoked whenever a node split occurs.
        The callback receives information about the split: the original leaf node being split,
        the new split node that replaced it, and the new child leaf nodes. This can be used
        to implement distributed training where split information is published to external
        systems like Kafka for consumption by inference processes.

    Notes
    -----
    A Hoeffding Tree [^1] is an incremental, anytime decision tree induction algorithm that is
    capable of learning from massive data streams, assuming that the distribution generating
    examples does not change over time. Hoeffding trees exploit the fact that a small sample can
    often be enough to choose an optimal splitting attribute. This idea is supported mathematically
    by the Hoeffding bound, which quantifies the number of observations (in our case, examples)
    needed to estimate some statistics within a prescribed precision (in our case, the goodness of
    an attribute).

    A theoretically appealing feature of Hoeffding Trees not shared by other incremental decision
    tree learners is that it has sound guarantees of performance. Using the Hoeffding bound one
    can show that its output is asymptotically nearly identical to that of a non-incremental
    learner using infinitely many examples. Implementation based on MOA [^2].

    References
    ----------

    [^1]: G. Hulten, L. Spencer, and P. Domingos. Mining time-changing data streams.
       In KDD’01, pages 97–106, San Francisco, CA, 2001. ACM Press.

    [^2]: Albert Bifet, Geoff Holmes, Richard Kirkby, Bernhard Pfahringer.
       MOA: Massive Online Analysis; Journal of Machine Learning Research 11: 1601-1604, 2010.

    Examples
    --------

    >>> from river.datasets import synth
    >>> from river import evaluate
    >>> from river import metrics
    >>> from river import tree

    >>> gen = synth.Agrawal(classification_function=0, seed=42)
    >>> # Take 1000 instances from the infinite data generator
    >>> dataset = iter(gen.take(1000))

    >>> model = tree.HoeffdingTreeClassifier(
    ...     grace_period=100,
    ...     delta=1e-5,
    ...     nominal_attributes=['elevel', 'car', 'zipcode']
    ... )

    >>> metric = metrics.Accuracy()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    Accuracy: 84.58%
    """

    _GINI_SPLIT = "gini"
    _INFO_GAIN_SPLIT = "info_gain"
    _HELLINGER_SPLIT = "hellinger"
    _VALID_SPLIT_CRITERIA = [_GINI_SPLIT, _INFO_GAIN_SPLIT, _HELLINGER_SPLIT]

    _MAJORITY_CLASS = "mc"
    _NAIVE_BAYES = "nb"
    _NAIVE_BAYES_ADAPTIVE = "nba"
    _VALID_LEAF_PREDICTION = [_MAJORITY_CLASS, _NAIVE_BAYES, _NAIVE_BAYES_ADAPTIVE]

    def __init__(
        self,
        grace_period: int = 200,
        max_depth: int | None = None,
        split_criterion: str = "info_gain",
        delta: float = 1e-7,
        tau: float = 0.05,
        leaf_prediction: str = "nba",
        nb_threshold: int = 0,
        nominal_attributes: list | None = None,
        splitter: Splitter | None = None,
        binary_split: bool = False,
        min_branch_fraction: float = 0.01,
        max_share_to_split: float = 0.99,
        max_size: float = 100.0,
        memory_estimate_period: int = 1000000,
        stop_mem_management: bool = False,
        remove_poor_attrs: bool = False,
        merit_preprune: bool = True,
        split_callback: callable | None = None,
    ):
        super().__init__(
            max_depth=max_depth,
            binary_split=binary_split,
            max_size=max_size,
            memory_estimate_period=memory_estimate_period,
            stop_mem_management=stop_mem_management,
            remove_poor_attrs=remove_poor_attrs,
            merit_preprune=merit_preprune,
        )
        self.grace_period = grace_period
        self.split_criterion = split_criterion
        self.delta = delta
        self.tau = tau
        self.leaf_prediction = leaf_prediction
        self.nb_threshold = nb_threshold
        self.nominal_attributes = nominal_attributes

        if splitter is None:
            self.splitter = GaussianSplitter()
        else:
            if not splitter.is_target_class:
                raise ValueError("The chosen splitter cannot be used in classification tasks.")
            self.splitter = splitter  # type: ignore

        self.min_branch_fraction = min_branch_fraction
        self.max_share_to_split = max_share_to_split
        self.split_callback = split_callback

        # To keep track of the observed classes
        self.classes: set = set()

    @property
    def _mutable_attributes(self):
        return {"grace_period", "delta", "tau"}

    @HoeffdingTree.split_criterion.setter  # type: ignore
    def split_criterion(self, split_criterion):
        if split_criterion not in self._VALID_SPLIT_CRITERIA:
            print(
                f"Invalid split_criterion option {split_criterion}', will use default '{self._INFO_GAIN_SPLIT}'"
            )
            self._split_criterion = self._INFO_GAIN_SPLIT
        else:
            self._split_criterion = split_criterion

    @HoeffdingTree.leaf_prediction.setter  # type: ignore
    def leaf_prediction(self, leaf_prediction):
        if leaf_prediction not in self._VALID_LEAF_PREDICTION:
            print(
                f"Invalid leaf_prediction option {leaf_prediction}', will use default '{self._NAIVE_BAYES_ADAPTIVE}'"
            )
            self._leaf_prediction = self._NAIVE_BAYES_ADAPTIVE
        else:
            self._leaf_prediction = leaf_prediction

    def _new_leaf(self, initial_stats=None, parent=None):
        if initial_stats is None:
            initial_stats = {}
        if parent is None:
            depth = 0
        else:
            depth = parent.depth + 1

        if self._leaf_prediction == self._MAJORITY_CLASS:
            return LeafMajorityClass(initial_stats, depth, self.splitter)
        elif self._leaf_prediction == self._NAIVE_BAYES:
            return LeafNaiveBayes(initial_stats, depth, self.splitter)
        else:  # Naives Bayes Adaptive (default)
            return LeafNaiveBayesAdaptive(initial_stats, depth, self.splitter)

    def _new_split_criterion(self):
        if self._split_criterion == self._GINI_SPLIT:
            split_criterion = GiniSplitCriterion(self.min_branch_fraction)
        elif self._split_criterion == self._INFO_GAIN_SPLIT:
            split_criterion = InfoGainSplitCriterion(self.min_branch_fraction)
        elif self._split_criterion == self._HELLINGER_SPLIT:
            split_criterion = HellingerDistanceCriterion(self.min_branch_fraction)
        else:
            split_criterion = InfoGainSplitCriterion(self.min_branch_fraction)

        return split_criterion

    def _attempt_to_split(self, leaf: HTLeaf, parent: DTBranch, parent_branch: int, **kwargs):
        """Attempt to split a leaf.

        If the samples seen so far are not from the same class then:

        1. Find split candidates and select the top 2.
        2. Compute the Hoeffding bound.
        3. If the difference between the top 2 split candidates is larger than the Hoeffding bound:
           3.1 Replace the leaf node by a split node (branch node).
           3.2 Add a new leaf node on each branch of the new split node.
           3.3 Update tree's metrics

        Optional: Disable poor attributes. Depends on the tree's configuration.

        Parameters
        ----------
        leaf
            The leaf to evaluate.
        parent
            The leaf's parent.
        parent_branch
            Parent leaf's branch index.
        kwargs
            Other parameters passed to the new branch.
        """
        print(f"\n   🔍 SPLIT ANALYSIS: Analyzing leaf for potential split")
        print(f"      Leaf stats: {getattr(leaf, 'stats', {})}")
        
        is_pure = leaf.observed_class_distribution_is_pure()
        print(f"      Class distribution is pure: {is_pure}")
        
        if not is_pure:  # type: ignore
            split_criterion = self._new_split_criterion()
            print(f"      Split criterion: {type(split_criterion).__name__}")

            best_split_suggestions = leaf.best_split_suggestions(split_criterion, self)
            best_split_suggestions.sort()
            
            print(f"      Found {len(best_split_suggestions)} split suggestions:")
            for i, suggestion in enumerate(best_split_suggestions):
                feature = getattr(suggestion, 'feature', 'unknown')
                merit = getattr(suggestion, 'merit', 'unknown')
                print(f"        {i+1}. Feature: {feature}, Merit: {merit}")
            
            should_split = False
            if len(best_split_suggestions) < 2:
                should_split = len(best_split_suggestions) > 0
                print(f"      Only {len(best_split_suggestions)} suggestion(s), split decision: {should_split}")
            else:
                hoeffding_bound = self._hoeffding_bound(
                    split_criterion.range_of_merit(leaf.stats),
                    self.delta,
                    leaf.total_weight,
                )
                best_suggestion = best_split_suggestions[-1]
                second_best_suggestion = best_split_suggestions[-2]
                merit_diff = best_suggestion.merit - second_best_suggestion.merit
                
                print(f"      Hoeffding bound calculation:")
                print(f"        Range of merit: {split_criterion.range_of_merit(leaf.stats)}")
                print(f"        Delta: {self.delta}, Leaf weight: {leaf.total_weight}")
                print(f"        Hoeffding bound: {hoeffding_bound}")
                print(f"        Best merit: {best_suggestion.merit}")
                print(f"        Second best merit: {second_best_suggestion.merit}")
                print(f"        Merit difference: {merit_diff}")
                print(f"        Tau threshold: {self.tau}")
                
                if (merit_diff > hoeffding_bound or hoeffding_bound < self.tau):
                    should_split = True
                    print(f"        ✅ SPLIT APPROVED: Merit diff > Hoeffding bound OR bound < tau")
                else:
                    print(f"        ❌ SPLIT REJECTED: Not enough evidence yet")
                if self.remove_poor_attrs:
                    poor_atts = set()
                    # Add any poor attribute to set
                    for suggestion in best_split_suggestions:
                        if (
                            suggestion.feature
                            and best_suggestion.merit - suggestion.merit > hoeffding_bound
                        ):
                            poor_atts.add(suggestion.feature)
                    for poor_att in poor_atts:
                        leaf.disable_attribute(poor_att)
            if should_split:
                split_decision = best_split_suggestions[-1]
                print(f"\n      🎯 EXECUTING SPLIT:")
                print(f"         Selected feature: {split_decision.feature}")
                print(f"         Numerical feature: {split_decision.numerical_feature}")
                print(f"         Multiway split: {split_decision.multiway_split}")
                
                # 🔍 DETAILED SPLIT DECISION INSPECTION
                print(f"\n         📋 SPLIT DECISION DETAILS:")
                print(f"         ===========================")
                print(f"            Type: {type(split_decision).__name__}")
                print(f"            Merit: {getattr(split_decision, 'merit', 'N/A')}")
                print(f"            Feature: {split_decision.feature}")
                print(f"            Numerical feature: {split_decision.numerical_feature}")
                print(f"            Multiway split: {split_decision.multiway_split}")
                
                # Check for threshold (numeric splits)
                if hasattr(split_decision, 'threshold'):
                    print(f"            Threshold: {split_decision.threshold}")
                
                # Check for children statistics
                if hasattr(split_decision, 'children_stats'):
                    print(f"            Children stats: {split_decision.children_stats}")
                    print(f"            Number of children: {len(split_decision.children_stats) if split_decision.children_stats else 0}")
                
                # Show all attributes of split_decision
                print(f"            All attributes: {[attr for attr in dir(split_decision) if not attr.startswith('_')]}")
                
                if split_decision.feature is None:
                    # Pre-pruning - null wins
                    print(f"         🚫 PRE-PRUNING: Null split wins, deactivating leaf")
                    leaf.deactivate()
                    self._n_inactive_leaves += 1
                    self._n_active_leaves -= 1
                else:
                    branch = self._branch_selector(
                        split_decision.numerical_feature, split_decision.multiway_split
                    )
                    print(f"         🌿 Branch type selected: {branch.__name__}")
                    
                    leaves = tuple(
                        self._new_leaf(initial_stats, parent=leaf)
                        for initial_stats in split_decision.children_stats  # type: ignore
                    )
                    print(f"         Created {len(leaves)} new leaf nodes")

                    new_split = split_decision.assemble(
                        branch, leaf.stats, leaf.depth, *leaves, **kwargs
                    )
                    print(f"         🔧 Assembled new split node: {type(new_split).__name__}")
                    
                    # 🔍 DETAILED NODE INSPECTION for distributed training
                    print(f"\n         📦 NODE CONTENTS INSPECTION:")
                    print(f"         ================================")
                    
                    # Original leaf being replaced
                    print(f"         🍃 ORIGINAL LEAF DATA:")
                    print(f"            Type: {type(leaf).__name__}")
                    print(f"            Depth: {getattr(leaf, 'depth', 'N/A')}")
                    print(f"            Total weight: {getattr(leaf, 'total_weight', 'N/A')}")
                    print(f"            Stats: {getattr(leaf, 'stats', {})}")
                    print(f"            Last split attempt: {getattr(leaf, 'last_split_attempt_at', 'N/A')}")
                    print(f"            Is active: {leaf.is_active() if hasattr(leaf, 'is_active') else 'N/A'}")
                    
                    # 🧠 NAIVE BAYES DATA FOR KAFKA
                    print(f"         🧠 NAIVE BAYES DATA FOR SERIALIZATION:")
                    if hasattr(leaf, '_mc_correct_weight'):
                        print(f"            MC correct weight: {leaf._mc_correct_weight}")
                    if hasattr(leaf, '_nb_correct_weight'):
                        print(f"            NB correct weight: {leaf._nb_correct_weight}")
                    
                    if hasattr(leaf, 'splitters') and leaf.splitters:
                        print(f"            📈 FEATURE MODELS FOR KAFKA:")
                        for attr, splitter in leaf.splitters.items():
                            print(f"              {attr}: {type(splitter).__name__}")
                            
                            # This is the data you'd need to serialize for full reconstruction
                            splitter_data = {}
                            if hasattr(splitter, '_var_per_class'):
                                splitter_data['variances'] = dict(splitter._var_per_class)
                            if hasattr(splitter, '_mean_per_class'):
                                splitter_data['means'] = dict(splitter._mean_per_class)
                            if hasattr(splitter, '_n_samples_per_class'):
                                splitter_data['samples'] = dict(splitter._n_samples_per_class)
                            if hasattr(splitter, '_counts'):
                                splitter_data['counts'] = dict(splitter._counts)
                            
                            if splitter_data:
                                print(f"                Serializable data: {splitter_data}")
                            else:
                                print(f"                No extractable parameters")
                    
                    # New split node
                    print(f"         🌿 NEW SPLIT NODE DATA:")
                    print(f"            Type: {type(new_split).__name__}")
                    print(f"            Feature: {getattr(new_split, 'feature', 'N/A')}")
                    print(f"            Depth: {getattr(new_split, 'depth', 'N/A')}")
                    print(f"            Stats: {getattr(new_split, 'stats', {})}")
                    if hasattr(new_split, 'threshold'):
                        print(f"            Threshold: {new_split.threshold}")
                        print(f"            🎯 SPLIT CONDITION: {new_split.feature} <= {new_split.threshold} (LEFT) | {new_split.feature} > {new_split.threshold} (RIGHT)")
                    if hasattr(new_split, 'children'):
                        print(f"            Number of children: {len(new_split.children)}")
                    print(f"            Max branches: {new_split.max_branches() if hasattr(new_split, 'max_branches') else 'N/A'}")
                    
                    # Show how to traverse this split
                    print(f"         🧭 HOW INFERENCE WORKS:")
                    if hasattr(new_split, 'threshold'):
                        print(f"            For new instance x:")
                        print(f"              if x['{new_split.feature}'] <= {new_split.threshold}:")
                        print(f"                  → go to LEFT child (branch 0)")
                        print(f"              else:")
                        print(f"                  → go to RIGHT child (branch 1)")
                    
                    # New leaf nodes
                    print(f"         🌱 NEW LEAF NODES DATA:")
                    for i, new_leaf in enumerate(leaves):
                        print(f"            Leaf {i}:")
                        print(f"              Type: {type(new_leaf).__name__}")
                        print(f"              Depth: {getattr(new_leaf, 'depth', 'N/A')}")
                        print(f"              Stats: {getattr(new_leaf, 'stats', {})}")
                        print(f"              Total weight: {getattr(new_leaf, 'total_weight', 'N/A')}")
                        print(f"              Parent: {type(getattr(new_leaf, 'parent', None)).__name__ if hasattr(new_leaf, 'parent') and new_leaf.parent else 'None'}")

                    self._n_active_leaves -= 1
                    self._n_active_leaves += len(leaves)
                    
                    if parent is None:
                        print(f"         🌳 REPLACING ROOT: Old leaf becomes new split node")
                        self._root = new_split
                    else:
                        print(f"         🔗 REPLACING CHILD: Updating parent's child[{parent_branch}]")
                        parent.children[parent_branch] = new_split
                    
                    print(f"         📊 Updated tree stats: active_leaves={self._n_active_leaves}, inactive_leaves={getattr(self, '_n_inactive_leaves', 0)}")

                    # Invoke the split callback if provided
                    if self.split_callback is not None:
                        print(f"         📡 CALLBACK: Invoking split callback for distributed training")
                        try:
                            callback_info = {
                                'split_type': 'node_split',
                                'original_leaf': leaf,
                                'new_split_node': new_split,
                                'new_leaves': leaves,
                                'split_feature': split_decision.feature,
                                'split_decision': split_decision,
                                'parent': parent,
                                'parent_branch': parent_branch,
                                'tree_id': id(self)  # Unique identifier for this tree instance
                            }
                            
                            print(f"         📦 CALLBACK DATA STRUCTURE FOR KAFKA:")
                            print(f"         =====================================")
                            print(f"            split_type: {callback_info['split_type']}")
                            print(f"            split_feature: {callback_info['split_feature']}")
                            print(f"            tree_id: {callback_info['tree_id']}")
                            print(f"            parent_branch: {callback_info['parent_branch']}")
                            
                            # Show serializable data from original leaf
                            print(f"            original_leaf_serializable_data:")
                            print(f"              type: {type(callback_info['original_leaf']).__name__}")
                            print(f"              depth: {getattr(callback_info['original_leaf'], 'depth', 'N/A')}")
                            print(f"              stats: {getattr(callback_info['original_leaf'], 'stats', {})}")
                            
                            # Show serializable data from new split node
                            print(f"            new_split_node_serializable_data:")
                            print(f"              type: {type(callback_info['new_split_node']).__name__}")
                            print(f"              feature: {getattr(callback_info['new_split_node'], 'feature', 'N/A')}")
                            print(f"              depth: {getattr(callback_info['new_split_node'], 'depth', 'N/A')}")
                            if hasattr(callback_info['new_split_node'], 'threshold'):
                                print(f"              threshold: {callback_info['new_split_node'].threshold}")
                            
                            # Show serializable data from new leaves
                            print(f"            new_leaves_serializable_data:")
                            for i, new_leaf in enumerate(callback_info['new_leaves']):
                                print(f"              leaf_{i}:")
                                print(f"                type: {type(new_leaf).__name__}")
                                print(f"                depth: {getattr(new_leaf, 'depth', 'N/A')}")
                                print(f"                stats: {getattr(new_leaf, 'stats', {})}")
                            
                            self.split_callback(callback_info)
                            print(f"         ✅ Callback executed successfully")
                        except Exception as e:
                            # Don't let callback errors break the training process
                            print(f"         ❌ Warning: split_callback failed with error: {e}")
                    else:
                        print(f"         📡 No split callback configured")

            else:
                print(f"      ❌ SPLIT REJECTED: Conditions not met")
                
            # Manage memory
            print(f"   🧹 MEMORY MANAGEMENT: Enforcing size limits")
            self._enforce_size_limit()

    def learn_one(self, x, y, *, w=1.0):
        """Train the model on instance x and corresponding target y.

        Parameters
        ----------
        x
            Instance attributes.
        y
            Class label for sample x.
        w
            Sample weight.

        Notes
        -----
        Training tasks:

        * If the tree is empty, create a leaf node as the root.
        * If the tree is already initialized, find the corresponding leaf for
          the instance and update the leaf node statistics.
        * If growth is allowed and the number of instances that the leaf has
          observed between split attempts exceed the grace period then attempt
          to split.
        """
        
        print(f"\n🌱 LEARNING: x={x}, y={y}, weight={w}")
        print(f"   Total training weight so far: {self._train_weight_seen_by_model}")

        # Updates the set of observed classes
        self.classes.add(y)
        print(f"   Known classes: {sorted(self.classes)}")

        self._train_weight_seen_by_model += w

        if self._root is None:
            self._root = self._new_leaf()
            self._n_active_leaves = 1
            print(f"   🌳 TREE INITIALIZATION: Created root leaf node")
            print(f"   📊 Tree stats: active_leaves={self._n_active_leaves}, inactive_leaves={getattr(self, '_n_inactive_leaves', 0)}")

        p_node = None
        node = None
        if isinstance(self._root, DTBranch):
            print(f"   🚶 TREE TRAVERSAL: Starting from split node (tree has structure)")
            path = iter(self._root.walk(x, until_leaf=False))
            depth = 0
            while True:
                aux = next(path, None)
                if aux is None:
                    break
                p_node = node
                node = aux
                if isinstance(node, DTBranch):
                    branch_desc = node.repr_split if hasattr(node, 'repr_split') else f"feature={getattr(node, 'feature', '?')}"
                    print(f"     Depth {depth}: Split node - {branch_desc}")
                    
                    # Show the actual split condition and decision
                    if hasattr(node, 'feature') and hasattr(node, 'threshold'):
                        feature_value = x.get(node.feature, 'MISSING')
                        if feature_value != 'MISSING':
                            goes_left = feature_value <= node.threshold
                            print(f"       🎯 SPLIT TEST: {node.feature}={feature_value} <= {node.threshold}? {goes_left}")
                            print(f"       🚶 DECISION: Going to {'LEFT' if goes_left else 'RIGHT'} child")
                        else:
                            print(f"       ⚠️  MISSING FEATURE: {node.feature} not in instance")
                    
                    # Show branch statistics if available
                    if hasattr(node, 'stats'):
                        print(f"       📊 Node stats: {node.stats}")
                else:
                    print(f"     Depth {depth}: Reached leaf node")
                depth += 1
        else:
            print(f"   🍃 SIMPLE CASE: Root is a leaf node")
            node = self._root

        if isinstance(node, HTLeaf):
            print(f"   📚 LEAF LEARNING: Updating leaf statistics")
            print(f"      Leaf depth: {getattr(node, 'depth', 'unknown')}")
            print(f"      Leaf weight before: {getattr(node, 'total_weight', 0)}")
            
            node.learn_one(x, y, w=w, tree=self)
            
            print(f"      Leaf weight after: {getattr(node, 'total_weight', 0)}")
            
            # 🔍 DETAILED LEAF STATE AFTER LEARNING
            print(f"      📊 DETAILED LEAF STATE:")
            print(f"         Stats after learning: {getattr(node, 'stats', {})}")
            if hasattr(node, 'stats') and node.stats:
                # Show class distribution if available
                print(f"         Class counts in stats: {dict(node.stats) if node.stats else 'Empty'}")
            
            # 🧠 NAIVE BAYES DATA INSPECTION
            print(f"      🧠 NAIVE BAYES DATA:")
            if hasattr(node, '_mc_correct_weight'):
                print(f"         Majority Class correct weight: {node._mc_correct_weight}")
            if hasattr(node, '_nb_correct_weight'):
                print(f"         Naive Bayes correct weight: {node._nb_correct_weight}")
            
            # Show splitters (the heart of Naive Bayes!)
            if hasattr(node, 'splitters') and node.splitters:
                print(f"         📈 ATTRIBUTE OBSERVERS (Naive Bayes Features):")
                for attr, splitter in node.splitters.items():
                    print(f"           Feature '{attr}': {type(splitter).__name__}")
                    
                    # Show the internal statistics of each splitter
                    if hasattr(splitter, 'cond_proba'):
                        print(f"             📊 Feature Statistics:")
                        # Try to show internal state
                        if hasattr(splitter, '_var_per_class'):
                            print(f"               Variances per class: {dict(splitter._var_per_class)}")
                        if hasattr(splitter, '_mean_per_class'):
                            print(f"               Means per class: {dict(splitter._mean_per_class)}")
                        if hasattr(splitter, '_n_samples_per_class'):
                            print(f"               Samples per class: {dict(splitter._n_samples_per_class)}")
                        
                        # For nominal splitters
                        if hasattr(splitter, '_counts'):
                            print(f"               Value counts: {dict(splitter._counts)}")
                        
                        # Show a sample conditional probability
                        if hasattr(node, 'stats') and node.stats:
                            for class_label in list(node.stats.keys())[:2]:  # Show first 2 classes
                                try:
                                    # Use the last seen value for this feature
                                    test_value = x.get(attr, 0)
                                    cond_prob = splitter.cond_proba(test_value, class_label)
                                    print(f"               P({attr}={test_value}|class={class_label}) = {cond_prob:.6f}")
                                except:
                                    print(f"               P({attr}|class={class_label}) = Cannot calculate")
            else:
                print(f"         No splitters available")
            
            print(f"      Growth allowed: {self._growth_allowed}")
            print(f"      Leaf active: {node.is_active()}")
            
            if self._growth_allowed and node.is_active():
                if node.depth >= self.max_depth:  # Max depth reached
                    print(f"   ⛔ MAX DEPTH REACHED: Deactivating leaf at depth {node.depth}")
                    node.deactivate()
                    self._n_active_leaves -= 1
                    self._n_inactive_leaves += 1
                else:
                    weight_seen = node.total_weight
                    weight_diff = weight_seen - node.last_split_attempt_at
                    print(f"   🤔 SPLIT CHECK: weight_seen={weight_seen}, last_attempt={node.last_split_attempt_at}")
                    print(f"      Weight difference: {weight_diff}, grace_period: {self.grace_period}")
                    
                    if weight_diff >= self.grace_period:
                        print(f"   🔥 SPLIT CONDITION MET: Attempting to split leaf!")
                        p_branch = p_node.branch_no(x) if isinstance(p_node, DTBranch) else None
                        self._attempt_to_split(node, p_node, p_branch)
                        node.last_split_attempt_at = weight_seen
                    else:
                        print(f"   ⏳ WAITING: Need {self.grace_period - weight_diff} more instances before split attempt")
        else:
            while True:
                # Split node encountered a previously unseen categorical value (in a multi-way
                #  test), so there is no branch to sort the instance to
                if node.max_branches() == -1 and node.feature in x:
                    # Create a new branch to the new categorical value
                    leaf = self._new_leaf(parent=node)
                    node.add_child(x[node.feature], leaf)
                    self._n_active_leaves += 1
                    node = leaf
                # The split feature is missing in the instance. Hence, we pass the new example
                # to the most traversed path in the current subtree
                else:
                    _, node = node.most_common_path()
                    # And we keep trying to reach a leaf
                    if isinstance(node, DTBranch):
                        node = node.traverse(x, until_leaf=False)
                # Once a leaf is reached, the traversal can stop
                if isinstance(node, HTLeaf):
                    break
            # Learn from the sample
            node.learn_one(x, y, w=w, tree=self)

        if self._train_weight_seen_by_model % self.memory_estimate_period == 0:
            print(f"   💾 MEMORY CHECK: Estimating model size (every {self.memory_estimate_period} instances)")
            self._estimate_model_size()
        
        print(f"   ✅ LEARNING COMPLETE for instance {int(self._train_weight_seen_by_model)}")
        print(f"   📊 Final tree stats: active_leaves={self._n_active_leaves}, inactive_leaves={getattr(self, '_n_inactive_leaves', 0)}")
        print(f"   🏗️  Tree depth: {getattr(self._root, 'depth', 0) if self._root else 0}")
        
        # Show current tree structure after learning
        if int(self._train_weight_seen_by_model) % 10 == 0:  # Show every 10 instances
            self.print_tree_structure()

    def predict_proba_one(self, x):
        print(f"\n🔮 PREDICTION: Predicting for x={x}")
        proba = {c: 0.0 for c in sorted(self.classes)}
        print(f"   Initial probabilities: {proba}")
        
        if self._root is not None:
            if isinstance(self._root, DTBranch):
                print(f"   🚶 TRAVERSING: Tree has structure, traversing to leaf")
                
                # Manual traversal to show each split condition
                current_node = self._root
                depth = 0
                print(f"   🗺️  PREDICTION PATH:")
                
                while isinstance(current_node, DTBranch):
                    # Show the split condition
                    if hasattr(current_node, 'feature') and hasattr(current_node, 'threshold'):
                        feature_value = x.get(current_node.feature, 'MISSING')
                        if feature_value != 'MISSING':
                            goes_left = feature_value <= current_node.threshold
                            print(f"     Depth {depth}: {current_node.feature}={feature_value} <= {current_node.threshold}? {goes_left}")
                            print(f"       → Taking {'LEFT' if goes_left else 'RIGHT'} branch")
                            
                            # Get the next node
                            branch_index = 0 if goes_left else 1
                            if hasattr(current_node, 'children') and len(current_node.children) > branch_index:
                                current_node = current_node.children[branch_index]
                                depth += 1
                            else:
                                print(f"       ⚠️  No child at branch {branch_index}")
                                break
                        else:
                            print(f"     Depth {depth}: Feature {current_node.feature} MISSING, using most common path")
                            _, current_node = current_node.most_common_path()
                            depth += 1
                    else:
                        print(f"     Depth {depth}: Non-standard split node")
                        break
                
                leaf = current_node
                print(f"   🍃 REACHED: Leaf node at depth {getattr(leaf, 'depth', depth)}")
            else:
                print(f"   🍃 SIMPLE: Root is a leaf node")
                leaf = self._root

            # 🧠 NAIVE BAYES PREDICTION ANALYSIS
            print(f"   🧠 NAIVE BAYES PREDICTION DETAILS:")
            if hasattr(leaf, '_mc_correct_weight') and hasattr(leaf, '_nb_correct_weight'):
                print(f"      MC correct weight: {leaf._mc_correct_weight}")
                print(f"      NB correct weight: {leaf._nb_correct_weight}")
                uses_nb = leaf._nb_correct_weight >= leaf._mc_correct_weight
                print(f"      Uses Naive Bayes: {uses_nb} ({'NB >= MC' if uses_nb else 'MC > NB'})")
            
            # Show the Naive Bayes calculation step by step
            if hasattr(leaf, 'stats') and hasattr(leaf, 'splitters') and leaf.splitters:
                print(f"      🔍 NAIVE BAYES CALCULATION:")
                
                # Manual Naive Bayes calculation for transparency
                total_weight = sum(leaf.stats.values()) if leaf.stats else 0
                print(f"         Total instances seen: {total_weight}")
                print(f"         Class priors:")
                
                nb_votes = {}
                for class_label, class_weight in (leaf.stats or {}).items():
                    if class_weight > 0:
                        prior = class_weight / total_weight
                        log_prior = __import__('math').log(prior)
                        nb_votes[class_label] = log_prior
                        print(f"           P(class={class_label}) = {class_weight}/{total_weight} = {prior:.4f} (log: {log_prior:.4f})")
                
                print(f"         Feature likelihoods:")
                for attr, value in x.items():
                    if attr in leaf.splitters:
                        splitter = leaf.splitters[attr]
                        print(f"           Feature '{attr}' = {value}:")
                        for class_label in (leaf.stats or {}).keys():
                            try:
                                likelihood = splitter.cond_proba(value, class_label)
                                log_likelihood = __import__('math').log(likelihood) if likelihood > 0 else float('-inf')
                                print(f"             P({attr}={value}|class={class_label}) = {likelihood:.6f} (log: {log_likelihood:.4f})")
                                if class_label in nb_votes:
                                    nb_votes[class_label] += log_likelihood
                            except Exception as e:
                                print(f"             P({attr}={value}|class={class_label}) = Error: {e}")
                
                print(f"         Final log-posteriors: {nb_votes}")
            
            leaf_prediction = leaf.prediction(x, tree=self)
            print(f"   📊 LEAF PREDICTION: {leaf_prediction}")
            proba.update(leaf_prediction)
            print(f"   🎯 FINAL PROBABILITIES: {proba}")
        else:
            print(f"   🚫 NO TREE: Root is None, returning default probabilities")
            
        return proba

    def print_tree_structure(self):
        """Print the complete tree structure with all split conditions."""
        print(f"\n🌳 COMPLETE TREE STRUCTURE:")
        print("=" * 50)
        
        if self._root is None:
            print("   Empty tree (no root)")
            return
        
        self._print_node(self._root, depth=0, prefix="")
    
    def _print_node(self, node, depth, prefix):
        """Recursively print tree nodes with split conditions."""
        indent = "  " * depth
        
        if isinstance(node, DTBranch):
            # Split node
            node_info = f"{prefix}{indent}🌿 SPLIT"
            if hasattr(node, 'feature'):
                node_info += f" on '{node.feature}'"
            if hasattr(node, 'threshold'):
                node_info += f" <= {node.threshold}"
            if hasattr(node, 'stats'):
                node_info += f" | Stats: {node.stats}"
            print(node_info)
            
            # Print children
            if hasattr(node, 'children'):
                for i, child in enumerate(node.children):
                    if child is not None:
                        branch_label = ""
                        if hasattr(node, 'threshold') and hasattr(node, 'feature'):
                            if i == 0:
                                branch_label = f"[{node.feature} <= {node.threshold}] "
                            else:
                                branch_label = f"[{node.feature} > {node.threshold}] "
                        else:
                            branch_label = f"[Branch {i}] "
                        
                        self._print_node(child, depth + 1, branch_label)
        else:
            # Leaf node
            leaf_info = f"{prefix}{indent}🍃 LEAF"
            if hasattr(node, 'stats'):
                leaf_info += f" | Stats: {node.stats}"
            if hasattr(node, 'total_weight'):
                leaf_info += f" | Weight: {node.total_weight}"
            print(leaf_info)

    def get_all_nodes(self):
        """Retrieve all nodes in the tree with their complete data including splitters."""
        print(f"\n📦 RETRIEVING ALL NODES WITH COMPLETE DATA:")
        print("=" * 60)
        
        if self._root is None:
            print("   Empty tree (no nodes)")
            return []
        
        all_nodes = []
        self._collect_all_nodes(self._root, all_nodes, depth=0, path="ROOT")
        return all_nodes
    
    def _collect_all_nodes(self, node, node_list, depth, path):
        """Recursively collect all nodes with their complete information."""
        
        # Create comprehensive node information
        node_info = {
            'path': path,
            'depth': depth,
            'node_type': type(node).__name__,
            'node_object': node,  # Keep reference to actual node
        }
        
        print(f"\n🔍 NODE ANALYSIS: {path} (Depth {depth})")
        print(f"   Type: {node_info['node_type']}")
        
        if isinstance(node, DTBranch):
            # SPLIT NODE ANALYSIS
            print(f"   🌿 SPLIT NODE DETAILS:")
            
            # Basic split information
            node_info.update({
                'is_split': True,
                'feature': getattr(node, 'feature', None),
                'threshold': getattr(node, 'threshold', None),
                'stats': getattr(node, 'stats', {}),
                'children_count': len(getattr(node, 'children', [])),
                'max_branches': getattr(node, 'max_branches', lambda: 0)(),
            })
            
            print(f"      Split feature: {node_info['feature']}")
            print(f"      Threshold: {node_info['threshold']}")
            print(f"      Stats: {node_info['stats']}")
            print(f"      Children count: {node_info['children_count']}")
            print(f"      Max branches: {node_info['max_branches']}")
            
            # Show split condition
            if node_info['feature'] and node_info['threshold'] is not None:
                print(f"      🎯 Split condition: {node_info['feature']} <= {node_info['threshold']}")
                print(f"         LEFT (≤): {node_info['feature']} <= {node_info['threshold']}")
                print(f"         RIGHT (>): {node_info['feature']} > {node_info['threshold']}")
            
            # Collect children
            if hasattr(node, 'children'):
                node_info['children'] = []
                for i, child in enumerate(node.children):
                    if child is not None:
                        child_path = f"{path}->CHILD[{i}]"
                        if node_info['feature'] and node_info['threshold'] is not None:
                            if i == 0:
                                child_path = f"{path}->LEFT({node_info['feature']}<={node_info['threshold']})"
                            else:
                                child_path = f"{path}->RIGHT({node_info['feature']}>{node_info['threshold']})"
                        
                        child_nodes = self._collect_all_nodes(child, node_list, depth + 1, child_path)
                        node_info['children'].extend(child_nodes)
        
        else:
            # LEAF NODE ANALYSIS  
            print(f"   🍃 LEAF NODE DETAILS:")
            
            # Basic leaf information
            node_info.update({
                'is_split': False,
                'stats': getattr(node, 'stats', {}),
                'total_weight': getattr(node, 'total_weight', 0),
                'is_active': getattr(node, 'is_active', lambda: False)(),
                'last_split_attempt_at': getattr(node, 'last_split_attempt_at', 0),
            })
            
            print(f"      Stats: {node_info['stats']}")
            print(f"      Total weight: {node_info['total_weight']}")
            print(f"      Is active: {node_info['is_active']}")
            print(f"      Last split attempt: {node_info['last_split_attempt_at']}")
            
            # NAIVE BAYES ANALYSIS
            print(f"      🧠 NAIVE BAYES DATA:")
            naive_bayes_data = {}
            
            # Performance tracking
            if hasattr(node, '_mc_correct_weight'):
                naive_bayes_data['mc_correct_weight'] = node._mc_correct_weight
                print(f"         MC correct weight: {node._mc_correct_weight}")
            
            if hasattr(node, '_nb_correct_weight'):
                naive_bayes_data['nb_correct_weight'] = node._nb_correct_weight
                print(f"         NB correct weight: {node._nb_correct_weight}")
            
            # Determine which prediction method is used
            if 'mc_correct_weight' in naive_bayes_data and 'nb_correct_weight' in naive_bayes_data:
                uses_nb = naive_bayes_data['nb_correct_weight'] >= naive_bayes_data['mc_correct_weight']
                naive_bayes_data['uses_naive_bayes'] = uses_nb
                print(f"         Uses Naive Bayes: {uses_nb} ({'NB' if uses_nb else 'MC'} prediction)")
            
            # SPLITTERS ANALYSIS (THE HEART OF NAIVE BAYES!)
            print(f"      📊 SPLITTERS (Feature Models):")
            splitters_data = {}
            
            if hasattr(node, 'splitters') and node.splitters:
                print(f"         Number of features: {len(node.splitters)}")
                
                for feature_name, splitter in node.splitters.items():
                    print(f"\n         🔬 FEATURE: '{feature_name}'")
                    print(f"            Splitter type: {type(splitter).__name__}")
                    
                    splitter_info = {
                        'type': type(splitter).__name__,
                        'feature_name': feature_name,
                        'splitter_object': splitter  # Keep reference for direct access
                    }
                    
                    # GAUSSIAN SPLITTER ANALYSIS
                    if hasattr(splitter, '_att_dist_per_class'):  # GaussianSplitter
                        print(f"            📈 GAUSSIAN STATISTICS:")
                        
                        # Collect all Gaussian parameters
                        gaussian_data = {}
                        
                        if hasattr(splitter, '_min_per_class'):
                            gaussian_data['min_per_class'] = dict(splitter._min_per_class)
                            print(f"               Min values: {gaussian_data['min_per_class']}")
                        
                        if hasattr(splitter, '_max_per_class'):
                            gaussian_data['max_per_class'] = dict(splitter._max_per_class)
                            print(f"               Max values: {gaussian_data['max_per_class']}")
                        
                        # Extract detailed Gaussian parameters
                        gaussian_distributions = {}
                        for class_label, gaussian_obj in splitter._att_dist_per_class.items():
                            print(f"               📊 Class {class_label}:")
                            
                            class_gaussian = {}
                            if hasattr(gaussian_obj, 'n_samples'):
                                class_gaussian['n_samples'] = gaussian_obj.n_samples
                                print(f"                  Samples: {gaussian_obj.n_samples}")
                            
                            if hasattr(gaussian_obj, 'mean'):
                                try:
                                    mean_val = gaussian_obj.mean.get() if hasattr(gaussian_obj.mean, 'get') else gaussian_obj.mean
                                    class_gaussian['mean'] = mean_val
                                    print(f"                  Mean: {mean_val}")
                                except:
                                    print(f"                  Mean: Cannot access")
                            
                            if hasattr(gaussian_obj, 'get'):  # Variance
                                try:
                                    var_val = gaussian_obj.get()
                                    class_gaussian['variance'] = var_val
                                    print(f"                  Variance: {var_val}")
                                except:
                                    print(f"                  Variance: Cannot access")
                            
                            # Try to access internal sum and sum_squares for reconstruction
                            if hasattr(gaussian_obj, '_sum'):
                                class_gaussian['sum'] = gaussian_obj._sum
                                print(f"                  Sum: {gaussian_obj._sum}")
                            if hasattr(gaussian_obj, '_sum_squares'):
                                class_gaussian['sum_squares'] = gaussian_obj._sum_squares  
                                print(f"                  Sum squares: {gaussian_obj._sum_squares}")
                            
                            gaussian_distributions[class_label] = class_gaussian
                        
                        gaussian_data['distributions'] = gaussian_distributions
                        splitter_info['gaussian_data'] = gaussian_data
                    
                    # NOMINAL SPLITTER ANALYSIS
                    elif hasattr(splitter, '_att_dist_per_class') and hasattr(splitter, '_att_values'):  # NominalSplitter
                        print(f"            📊 NOMINAL STATISTICS:")
                        
                        nominal_data = {}
                        
                        if hasattr(splitter, '_total_weight_observed'):
                            nominal_data['total_weight'] = splitter._total_weight_observed
                            print(f"               Total weight: {nominal_data['total_weight']}")
                        
                        if hasattr(splitter, '_att_values'):
                            nominal_data['unique_values'] = list(splitter._att_values)
                            print(f"               Unique values: {nominal_data['unique_values']}")
                        
                        if hasattr(splitter, '_att_dist_per_class'):
                            nominal_data['class_distributions'] = dict(splitter._att_dist_per_class)
                            print(f"               Class distributions:")
                            for class_label, value_counts in splitter._att_dist_per_class.items():
                                print(f"                  Class {class_label}: {dict(value_counts)}")
                        
                        splitter_info['nominal_data'] = nominal_data
                    
                    # Test conditional probabilities
                    print(f"            🎯 SAMPLE CONDITIONAL PROBABILITIES:")
                    if hasattr(node, 'stats') and node.stats:
                        for class_label in list(node.stats.keys())[:2]:  # Test first 2 classes
                            try:
                                # Use a sample value
                                if hasattr(splitter, '_att_values') and splitter._att_values:
                                    # Nominal - use first unique value
                                    test_value = list(splitter._att_values)[0]
                                elif hasattr(splitter, '_min_per_class') and splitter._min_per_class:
                                    # Gaussian - use mean of first class
                                    test_value = list(splitter._min_per_class.values())[0] + 1
                                else:
                                    test_value = 1.0
                                
                                cond_prob = splitter.cond_proba(test_value, class_label)
                                print(f"               P({feature_name}={test_value}|class={class_label}) = {cond_prob:.8f}")
                            except Exception as e:
                                print(f"               P({feature_name}|class={class_label}) = Error: {e}")
                    
                    splitters_data[feature_name] = splitter_info
            else:
                print(f"         No splitters available")
            
            node_info['naive_bayes_data'] = naive_bayes_data
            node_info['splitters_data'] = splitters_data
        
        node_list.append(node_info)
        return [node_info]
    
    def get_node_by_path(self, target_path):
        """Get a specific node by its path for detailed inspection."""
        all_nodes = self.get_all_nodes()
        
        for node_info in all_nodes:
            if node_info['path'] == target_path:
                return node_info
        
        return None
    
    def get_all_splitters(self):
        """Get all splitters from all leaf nodes in the tree."""
        print(f"\n🧠 COLLECTING ALL SPLITTERS FROM TREE:")
        print("=" * 50)
        
        all_nodes = self.get_all_nodes()
        all_splitters = {}
        
        for node_info in all_nodes:
            if not node_info['is_split'] and 'splitters_data' in node_info:
                node_path = node_info['path']
                print(f"\n🍃 Leaf: {node_path}")
                
                for feature_name, splitter_info in node_info['splitters_data'].items():
                    splitter_key = f"{node_path}::{feature_name}"
                    all_splitters[splitter_key] = splitter_info
                    print(f"   📊 {feature_name}: {splitter_info['type']}")
        
        print(f"\n📈 TOTAL SPLITTERS FOUND: {len(all_splitters)}")
        return all_splitters

    def inspect_node_splitters(self, node_path="ROOT"):
        """Detailed inspection of splitters in a specific node."""
        print(f"\n🔬 DETAILED SPLITTER INSPECTION: {node_path}")
        print("=" * 60)
        
        node_info = self.get_node_by_path(node_path)
        
        if node_info is None:
            print(f"❌ Node not found: {node_path}")
            return
        
        if node_info['is_split']:
            print(f"🌿 This is a split node, no splitters available")
            return
        
        if 'splitters_data' not in node_info or not node_info['splitters_data']:
            print(f"🍃 This leaf has no splitters yet")
            return
        
        print(f"🍃 Leaf node with {len(node_info['splitters_data'])} splitters:")
        
        for feature_name, splitter_info in node_info['splitters_data'].items():
            print(f"\n   🔬 FEATURE: {feature_name}")
            print(f"      Type: {splitter_info['type']}")
            
            # Access the actual splitter object for live data
            if 'splitter_object' in splitter_info:
                live_splitter = splitter_info['splitter_object']
                print(f"      Live splitter object: {type(live_splitter).__name__}")
                
                # Show all accessible attributes
                print(f"      Accessible attributes:")
                for attr_name in sorted(dir(live_splitter)):
                    if not attr_name.startswith('__'):
                        try:
                            attr_value = getattr(live_splitter, attr_name)
                            if not callable(attr_value):
                                print(f"         {attr_name}: {attr_value}")
                        except:
                            print(f"         {attr_name}: <cannot access>")
        
        return node_info

    @property
    def _multiclass(self):
        return True
