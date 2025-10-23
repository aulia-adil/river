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
        leaf_update_callback: callable | None = None,
        leaf_update_threshold: int = 50,
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
        
        # Dual gate notification system
        self.split_callback = split_callback              # Gate 1: Split notifications
        self.leaf_update_callback = leaf_update_callback  # Gate 2: Leaf update notifications
        self.leaf_update_threshold = leaf_update_threshold # Customizable threshold (n)

        # To keep track of the observed classes
        self.classes: set = set()
        
        # Node ID system for O(1) lookup capability
        self._next_node_id = 0
        self._node_registry = {}  # Maps node_id -> node object

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

    def _generate_node_id(self):
        """Generate a unique node ID."""
        node_id = self._next_node_id
        self._next_node_id += 1
        return node_id
    
    def _register_node(self, node, node_id=None):
        """Register a node in the node registry for O(1) lookup."""
        if node_id is None:
            node_id = self._generate_node_id()
        
        # Assign the ID to the node
        node.node_id = node_id
        
        # Register in the lookup table
        self._node_registry[node_id] = node
        
        print(f"   🏷️  REGISTERED NODE: ID={node_id}, Type={type(node).__name__}")
        return node_id
    
    def _check_leaf_update_gate(self, node, previous_weight, current_weight):
        """🚪 GATE 2: Check if leaf has reached multiple of threshold instances."""
        
        if self.leaf_update_callback is None:
            return  # No callback configured
        
        # Check if we've crossed a multiple threshold
        previous_multiple = int(previous_weight // self.leaf_update_threshold)
        current_multiple = int(current_weight // self.leaf_update_threshold)
        print(f"node {getattr(node, 'node_id', 'unknown')}, previous_multiple: {previous_multiple}, current_multiple: {current_multiple}")
        
        if current_multiple > previous_multiple:
            # We've crossed a threshold!
            node_id = getattr(node, 'node_id', 'unknown')
            
            print(f"\n🚪 GATE 2 TRIGGERED: Leaf {node_id} reached {current_multiple * self.leaf_update_threshold} instances")
            print(f"   Previous weight: {previous_weight:.1f} → Current weight: {current_weight:.1f}")
            print(f"   Threshold: {self.leaf_update_threshold} (multiple #{current_multiple})")
            
            # Use create_update_payload to get the node state
            update_payload = self.create_update_payload(node_id, update_type='complete_node')
            
            if update_payload is None:
                print(f"   ❌ GATE 2 ERROR: Could not create update payload for node {node_id}")
                return
            try:
                self.leaf_update_callback(update_payload)
                print(f"   ✅ GATE 2 CALLBACK: Executed successfully")
            except Exception as e:
                print(f"   ❌ GATE 2 CALLBACK ERROR: {e}")
    
    def _extract_splitter_data_for_callback(self, leaf):
        """Extract splitter data from leaf for callback notifications."""
        splitters_data = {}
        
        if not hasattr(leaf, 'splitters') or not leaf.splitters:
            return splitters_data
        
        for feature_name, splitter in leaf.splitters.items():
            splitter_info = {
                "type": type(splitter).__name__,
                "feature_name": feature_name
            }

            print(f"      splitter: {splitter}")
            
            if hasattr(splitter, '_att_dist_per_class'):
                print(f"      splitter._att_dist_per_class: {splitter._att_dist_per_class}")
                
                # Check if this is a Gaussian splitter by examining the structure
                # Gaussian: class -> distribution object with methods
                # Nominal: class -> dict with category -> count
                is_gaussian_splitter = False
                is_nominal_splitter = False
                
                if splitter._att_dist_per_class:
                    # Get the first class distribution to check its type
                    first_class_dist = next(iter(splitter._att_dist_per_class.values()))
                    
                    # If it's a dict with string/category keys, it's nominal
                    if isinstance(first_class_dist, dict):
                        is_nominal_splitter = True
                        print(f"      → Detected NOMINAL splitter for {feature_name}")
                    # If it has methods like 'mean' or 'get', it's Gaussian (check for common Gaussian attributes)
                    elif (hasattr(first_class_dist, 'mean') or hasattr(first_class_dist, 'get') or 
                          hasattr(first_class_dist, 'n_samples') or 'Gaussian' in str(type(first_class_dist))):
                        is_gaussian_splitter = True
                        print(f"      → Detected GAUSSIAN splitter for {feature_name}")
                    else:
                        # Fallback: check splitter type name
                        splitter_type_name = type(splitter).__name__
                        if 'Gaussian' in splitter_type_name:
                            is_gaussian_splitter = True
                            print(f"      → Detected GAUSSIAN splitter for {feature_name} (by splitter type)")
                        elif 'Nominal' in splitter_type_name:
                            is_nominal_splitter = True
                            print(f"      → Detected NOMINAL splitter for {feature_name} (by splitter type)")
                        else:
                            print(f"      → Unknown splitter type for {feature_name}: {type(first_class_dist)} (splitter: {splitter_type_name})")
                
                if is_gaussian_splitter:
                    # Gaussian splitter data
                    gaussian_data = {}
                    
                    # Min/max per class
                    if hasattr(splitter, '_min_per_class'):
                        gaussian_data['min_per_class'] = dict(splitter._min_per_class)
                    if hasattr(splitter, '_max_per_class'):
                        gaussian_data['max_per_class'] = dict(splitter._max_per_class)
                    
                    # Distribution parameters per class
                    distributions = {}
                    for class_label, dist_obj in splitter._att_dist_per_class.items():
                        print(f"      dist_obj for class {class_label}: {dist_obj}")
                        class_data = {}
                        if hasattr(dist_obj, 'n_samples'):
                            class_data['n_samples'] = dist_obj.n_samples
                        if hasattr(dist_obj, 'mu'):
                            class_data['mu'] = dist_obj.mu
                        if hasattr(dist_obj, 'sigma'):
                            class_data['sigma'] = dist_obj.sigma
                        distributions[str(class_label)] = class_data
                    
                    gaussian_data['distributions'] = distributions
                    splitter_info['gaussian_data'] = gaussian_data
                
                elif is_nominal_splitter:
                    # Nominal splitter data
                    nominal_data = {}
                    if hasattr(splitter, '_total_weight_observed'):
                        nominal_data['total_weight'] = splitter._total_weight_observed
                    if hasattr(splitter, '_att_values'):
                        nominal_data['unique_values'] = list(splitter._att_values)
                    
                    # For nominal: _att_dist_per_class is {class: {category: count}}
                    nominal_data['class_distributions'] = {
                        str(k): dict(v) for k, v in splitter._att_dist_per_class.items()
                    }
                    splitter_info['nominal_data'] = nominal_data
            
            # Fallback: Check for nominal attributes without _att_dist_per_class
            elif hasattr(splitter, '_att_values'):
                nominal_data = {}
                if hasattr(splitter, '_total_weight_observed'):
                    nominal_data['total_weight'] = splitter._total_weight_observed
                if hasattr(splitter, '_att_values'):
                    nominal_data['unique_values'] = list(splitter._att_values)
                splitter_info['nominal_data'] = nominal_data
                print(f"      → Detected NOMINAL splitter (no _att_dist_per_class) for {feature_name}")
            
            splitters_data[feature_name] = splitter_info
        
        return splitters_data

    def _new_leaf(self, initial_stats=None, parent=None):
        if initial_stats is None:
            initial_stats = {}
        if parent is None:
            depth = 0
        else:
            depth = parent.depth + 1

        if self._leaf_prediction == self._MAJORITY_CLASS:
            leaf = LeafMajorityClass(initial_stats, depth, self.splitter)
        elif self._leaf_prediction == self._NAIVE_BAYES:
            leaf = LeafNaiveBayes(initial_stats, depth, self.splitter)
        else:  # Naives Bayes Adaptive (default)
            leaf = LeafNaiveBayesAdaptive(initial_stats, depth, self.splitter)
        
        # Assign unique ID and register the leaf
        self._register_node(leaf)
        return leaf

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
        if not leaf.observed_class_distribution_is_pure():  # type: ignore
            split_criterion = self._new_split_criterion()

            best_split_suggestions = leaf.best_split_suggestions(split_criterion, self)
            best_split_suggestions.sort()
            should_split = False
            if len(best_split_suggestions) < 2:
                should_split = len(best_split_suggestions) > 0
            else:
                hoeffding_bound = self._hoeffding_bound(
                    split_criterion.range_of_merit(leaf.stats),
                    self.delta,
                    leaf.total_weight,
                )
                best_suggestion = best_split_suggestions[-1]
                second_best_suggestion = best_split_suggestions[-2]
                if (
                    best_suggestion.merit - second_best_suggestion.merit > hoeffding_bound
                    or hoeffding_bound < self.tau
                ):
                    should_split = True
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
                if split_decision.feature is None:
                    # Pre-pruning - null wins
                    leaf.deactivate()
                    self._n_inactive_leaves += 1
                    self._n_active_leaves -= 1
                else:
                    branch = self._branch_selector(
                        split_decision.numerical_feature, split_decision.multiway_split
                    )
                    # print split_decision.children_stats
                    print(f"   📊 Split decision children stats: {split_decision.children_stats}")
                    leaves = tuple(
                        self._new_leaf(initial_stats, parent=leaf)
                        for initial_stats in split_decision.children_stats  # type: ignore
                    )

                    new_split = split_decision.assemble(
                        branch, leaf.stats, leaf.depth, *leaves, **kwargs
                    )

                    # Register the new split node with unique ID
                    self._register_node(new_split)
                    
                    # Remove the old leaf from registry since it's being replaced
                    if hasattr(leaf, 'node_id'):
                        self.remove_node_from_registry(leaf.node_id)

                    self._n_active_leaves -= 1
                    self._n_active_leaves += len(leaves)
                    if parent is None:
                        self._root = new_split
                    else:
                        parent.children[parent_branch] = new_split
                    
                    # Print registry status for debugging
                    print(f"   📊 Node registry size: {self.get_node_registry_size()}")
                    print(f"      Split created: ID={new_split.node_id}, Feature={split_decision.feature}")
                    print(f"      New leaves: {[leaf.node_id for leaf in leaves if hasattr(leaf, 'node_id')]}")
                    
                    # Invoke split callback if provided
                    if self.split_callback is not None:
                        split_info = {
                            'original_leaf': leaf,
                            'new_split_node': new_split,
                            'new_leaves': leaves,
                            'split_feature': split_decision.feature,
                            'parent': parent,
                            'parent_branch': parent_branch
                        }
                        try:
                            self.split_callback(split_info)
                        except Exception as e:
                            print(f"   ⚠️  Split callback error: {e}")

                # Manage memory
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

        # Updates the set of observed classes
        self.classes.add(y)

        self._train_weight_seen_by_model += w

        if self._root is None:
            self._root = self._new_leaf()
            self._n_active_leaves = 1

        p_node = None
        node = None
        if isinstance(self._root, DTBranch):
            path = iter(self._root.walk(x, until_leaf=False))
            while True:
                aux = next(path, None)
                if aux is None:
                    break
                p_node = node
                node = aux
        else:
            node = self._root

        if isinstance(node, HTLeaf):
            # Store previous weight for gate checking
            previous_weight = node.total_weight
            
            # Learn from the instance
            node.learn_one(x, y, w=w, tree=self)
            
            # Track whether a split will occur
            split_occurred = False
            
            if self._growth_allowed and node.is_active():
                if node.depth >= self.max_depth:  # Max depth reached
                    node.deactivate()
                    self._n_active_leaves -= 1
                    self._n_inactive_leaves += 1
                else:
                    weight_seen = node.total_weight
                    weight_diff = weight_seen - node.last_split_attempt_at
                    if weight_diff >= self.grace_period:
                        p_branch = p_node.branch_no(x) if isinstance(p_node, DTBranch) else None
                        
                        # 🚪 GATE 1: Split attempt - this will trigger split callback if split occurs
                        print(f"\n🚪 GATE 1 CHECK: Attempting split on node {getattr(node, 'node_id', 'unknown')}")
                        registry_size_before = self.get_node_registry_size()
                        
                        self._attempt_to_split(node, p_node, p_branch)
                        node.last_split_attempt_at = weight_seen
                        
                        # Check if split actually occurred
                        registry_size_after = self.get_node_registry_size()
                        if registry_size_after > registry_size_before:
                            split_occurred = True
                            print(f"✅ GATE 1 TRIGGERED: Split occurred! Registry: {registry_size_before} → {registry_size_after} nodes")
            
            # 🚪 GATE 2: Only trigger leaf update if NO split occurred
            # Rationale: If a split happened, the leaf is replaced and no longer exists.
            # The split callback provides complete information about the structural change.
            current_weight = node.total_weight
            if not split_occurred:
                self._check_leaf_update_gate(node, previous_weight, current_weight)
            else:
                print(f"⏭️  GATE 2 SKIPPED: Split occurred, leaf update callback not needed (node replaced)")
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
            # Store previous weight for gate checking
            previous_weight = node.total_weight if hasattr(node, 'total_weight') else 0
            
            node.learn_one(x, y, w=w, tree=self)
            
            # 🚪 GATE 2: Check for leaf update threshold notification
            current_weight = node.total_weight if hasattr(node, 'total_weight') else 0
            self._check_leaf_update_gate(node, previous_weight, current_weight)

        if self._train_weight_seen_by_model % self.memory_estimate_period == 0:
            self._estimate_model_size()

    def predict_proba_one(self, x):
        proba = {c: 0.0 for c in sorted(self.classes)}
        if self._root is not None:
            if isinstance(self._root, DTBranch):
                leaf = self._root.traverse(x, until_leaf=True)
            else:
                leaf = self._root

            proba.update(leaf.prediction(x, tree=self))
        return proba

    @property
    def _multiclass(self):
        return True
    
    # Node ID System Methods
    def get_node_by_id(self, node_id):
        """Get a node by its unique ID - O(1) lookup."""
        return self._node_registry.get(node_id, None)
    
    def get_all_node_ids(self):
        """Get all registered node IDs."""
        return list(self._node_registry.keys())
    
    def get_node_registry_size(self):
        """Get the number of nodes in the registry."""
        return len(self._node_registry)
    
    def print_node_registry(self):
        """Print all nodes in the registry with their IDs."""
        print(f"\n📋 NODE REGISTRY ({len(self._node_registry)} nodes):")
        print("=" * 50)
        
        for node_id, node in sorted(self._node_registry.items()):
            node_type = type(node).__name__
            depth = getattr(node, 'depth', 'N/A')
            stats = getattr(node, 'stats', {})
            
            if hasattr(node, 'feature'):  # Split node
                feature = getattr(node, 'feature', 'N/A')
                threshold = getattr(node, 'threshold', 'N/A')
                print(f"   ID {node_id:3d}: {node_type} | Depth: {depth} | Split: {feature} <= {threshold}")
            else:  # Leaf node
                total_weight = getattr(node, 'total_weight', 0)
                print(f"   ID {node_id:3d}: {node_type} | Depth: {depth} | Weight: {total_weight} | Stats: {stats}")
    
    def remove_node_from_registry(self, node_id):
        """Remove a node from the registry (useful for memory management)."""
        if node_id in self._node_registry:
            node = self._node_registry.pop(node_id)
            print(f"   🗑️  REMOVED NODE: ID={node_id}, Type={type(node).__name__}")
            return node
        return None
    
    def update_node_in_registry(self, node_id, updated_data):
        """Update node data and notify about the change for distributed systems."""
        node = self.get_node_by_id(node_id)
        if node is not None:
            print(f"   🔄 UPDATE NODE: ID={node_id}, Type={type(node).__name__}")
            print(f"      Updated data: {updated_data}")
            
            # This is where you could trigger Kafka notifications
            update_info = {
                'node_id': node_id,
                'node_type': type(node).__name__,
                'update_type': 'node_data_update',
                'updated_data': updated_data,
                'timestamp': __import__('time').time()
            }
            
            # You can add your Kafka callback here
            print(f"      📡 Ready for Kafka: {update_info}")
            return True
        return False
    
    def apply_distributed_update(self, update_payload):
        """Apply updates from distributed training processes to a specific node.
        
        This method allows nodes to receive and apply updates from other distributed 
        training processes, enabling synchronized learning across multiple instances.
        
        Parameters
        ----------
        node_id : int
            The ID of the node to update
        update_payload : dict
            Dictionary containing the update information with the following structure:
            {
                'update_type': 'leaf_stats' | 'splitter_data' | 'naive_bayes_data' | 'complete_node',
                'data': {
                    # Update-specific data
                }
            }
        
        Returns
        -------
        bool
            True if update was successfully applied, False otherwise
        """
        node_id = update_payload.get('node_id')
        node = self.get_node_by_id(node_id)
        if node is None:
            print(f"❌ Node {node_id} not found in registry")
            return False
        
        update_type = update_payload.get('update_type')
        update_data = update_payload.get('data', {})
        
        print(f"📡 APPLYING DISTRIBUTED UPDATE:")
        print(f"   Node ID: {node_id} ({type(node).__name__})")
        print(f"   Update type: {update_type}")
        
        try:
            if update_type == 'leaf_stats':
                return self._apply_leaf_stats_update(node, update_data)
            
            elif update_type == 'splitter_data':
                return self._apply_splitter_data_update(node, update_data)
            
            elif update_type == 'naive_bayes_data':
                return self._apply_naive_bayes_update(node, update_data)
            
            elif update_type == 'complete_node':
                return self._apply_complete_node_update(node, update_data)
            
            elif update_type == 'incremental_stats':
                return self._apply_incremental_stats_update(node, update_data)
            
            else:
                print(f"❌ Unknown update type: {update_type}")
                return False
                
        except Exception as e:
            print(f"❌ Error applying update: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _apply_leaf_stats_update(self, node, update_data):
        """Apply leaf statistics updates (class counts, weights)."""
        print(f"   🍃 Applying leaf stats update...")
        
        # Update class statistics
        if 'stats' in update_data:
            new_stats = update_data['stats']
            print(f"update_data: {update_data}")
            print(f"      Current stats: {dict(getattr(node, 'stats', {}))}")
            print(f"      New stats to sync: {new_stats}")
            
            # SYNCHRONIZE stats (replace, don't accumulate)
            if hasattr(node, 'stats'):
                # Clear existing stats and replace with new ones
                node.stats.clear()
                for class_label, count in new_stats.items():
                    # Convert string keys back to their original type (usually int)
                    # This handles JSON serialization where int keys become strings
                    try:
                        if isinstance(class_label, str) and class_label.lstrip('-').replace('.', '', 1).isdigit():
                            # Try int first, then float
                            if '.' in class_label:
                                class_key = float(class_label)
                            else:
                                class_key = int(class_label)
                        else:
                            class_key = class_label
                    except (ValueError, AttributeError):
                        class_key = class_label
                    
                    node.stats[class_key] = count
            else:
                # Convert keys when creating new stats dict
                converted_stats = {}
                for class_label, count in new_stats.items():
                    try:
                        if isinstance(class_label, str) and class_label.lstrip('-').replace('.', '', 1).isdigit():
                            if '.' in class_label:
                                class_key = float(class_label)
                            else:
                                class_key = int(class_label)
                        else:
                            class_key = class_label
                    except (ValueError, AttributeError):
                        class_key = class_label
                    converted_stats[class_key] = count
                node.stats = converted_stats
            
            print(f"      Synchronized stats: {dict(node.stats)}")
        
        # Note: total_weight is calculated from stats, so it's updated automatically
        # when we update the stats above
        if 'total_weight' in update_data:
            expected_weight = update_data['total_weight']
            actual_weight = getattr(node, 'total_weight', 0)
            print(f"      Expected weight: {expected_weight}")
            print(f"      Actual weight after sync: {actual_weight}")
            
            if abs(actual_weight - expected_weight) > 0.01:
                print(f"      ⚠️  Weight mismatch detected!")
            else:
                print(f"      ✅ Weight synchronized correctly")
        
        return True
    
    def _apply_splitter_data_update(self, node, update_data):
        """Apply splitter data updates (_att_dist_per_class, etc.)."""
        print(f"   🔀 Applying splitter data update...")
        
        if not hasattr(node, 'splitters'):
            print(f"      ⚠️ Node has no splitters attribute")
            return False
        
        # Ensure splitters dict exists (it might be None if leaf was deactivated)
        if node.splitters is None:
            print(f"      ℹ️  Node splitters is None, activating node...")
            node.activate()
        
        splitters_updates = update_data.get('splitters', {})
        
        for feature_name, splitter_update in splitters_updates.items():
            # Create splitter if it doesn't exist
            if feature_name not in node.splitters:
                print(f"      📝 Creating new splitter for feature: {feature_name}")
                
                # Determine splitter type from the update data
                if 'gaussian_data' in splitter_update:
                    # Create Gaussian splitter
                    splitter = self.splitter.clone()
                elif 'nominal_data' in splitter_update:
                    # Create Nominal splitter
                    splitter = node.new_nominal_splitter()
                else:
                    print(f"      ⚠️ Unknown splitter type for feature {feature_name}")
                    continue
                print(f"feature_name: {feature_name}")
                node.splitters[feature_name] = splitter
            else:
                splitter = node.splitters[feature_name]
            
            print(f"      📊 Updating splitter for feature: {feature_name}")
            
            # Update Gaussian splitter data
            if 'gaussian_data' in splitter_update:
                gaussian_data = splitter_update['gaussian_data']
                print(f"         🔢 Updating Gaussian splitter...")
                
                # Update _att_dist_per_class for Gaussian
                if 'distributions' in gaussian_data:
                    if not hasattr(splitter, '_att_dist_per_class'):
                        print(f"         ⚠️ Splitter has no _att_dist_per_class")
                        continue
                    
                    # Import required classes
                    from river.proba import Gaussian
                    from river import stats
                    
                    for class_label, class_data in gaussian_data['distributions'].items():
                        # Handle both string and numeric class labels
                        try:
                            if isinstance(class_label, str) and class_label.isdigit():
                                class_key = int(class_label)
                            elif isinstance(class_label, (int, float)):
                                class_key = class_label
                            else:
                                class_key = float(class_label) if str(class_label).replace('.', '').isdigit() else class_label
                        except (ValueError, AttributeError):
                            class_key = class_label
                        
                        # Get target parameters
                        target_mu = class_data.get('mu', 0.0)
                        target_sigma = class_data.get('sigma', 1.0)
                        target_n = class_data.get('n_samples', 1.0)
                        
                        # Calculate variance from sigma (var = sigma^2)
                        target_var = target_sigma ** 2
                        
                        # Create or update Gaussian distribution
                        if class_key in splitter._att_dist_per_class:
                            dist_obj = splitter._att_dist_per_class[class_key]
                            
                            # For Gaussian distributions, use _from_state for exact synchronization
                            if hasattr(dist_obj, 'mu') and hasattr(dist_obj, 'sigma'):
                                print(f"            🔄 Synchronizing Gaussian distribution for class {class_key}")
                                
                                # Use _from_state to create Gaussian with EXACT parameters
                                # Gaussian._from_state(n, m, sig, ddof)
                                # where: n=sample count, m=mean, sig=VARIANCE (not sum!), ddof=degrees of freedom
                                new_gaussian = Gaussian._from_state(
                                    n=target_n,
                                    m=target_mu,
                                    sig=target_var,  # sig is the variance
                                    ddof=1
                                )
                                
                                # Replace the distribution
                                splitter._att_dist_per_class[class_key] = new_gaussian
                                print(f"            ✅ Synced Gaussian: μ={new_gaussian.mu:.6f}, σ={new_gaussian.sigma:.6f}, n={new_gaussian.n_samples}")
                                print(f"               (target: μ={target_mu:.6f}, σ={target_sigma:.6f}, n={target_n})")
                            else:
                                # For non-Gaussian distributions, try incremental updates
                                if 'n_samples' in class_data and hasattr(dist_obj, 'update'):
                                    # Try incremental updates for other types
                                    for _ in range(int(class_data.get('n_samples', 0))):
                                        if 'mean' in class_data:
                                            dist_obj.update(class_data['mean'])
                        else:
                            # Class doesn't exist yet - create new Gaussian distribution
                            print(f"            ✨ Creating new Gaussian distribution for class {class_key}")
                            new_gaussian = Gaussian._from_state(
                                n=target_n,
                                m=target_mu,
                                sig=target_var,
                                ddof=1
                            )
                            splitter._att_dist_per_class[class_key] = new_gaussian
                            print(f"            ✅ Created Gaussian: μ={new_gaussian.mu:.6f}, σ={new_gaussian.sigma:.6f}, n={new_gaussian.n_samples}")
                        
                        print(f"         ✅ Updated class {class_key} distribution")
                
                # Update min/max per class
                if 'min_per_class' in gaussian_data and hasattr(splitter, '_min_per_class'):
                    for class_label, min_val in gaussian_data['min_per_class'].items():
                        # Handle both string and numeric class labels
                        try:
                            if isinstance(class_label, str) and class_label.isdigit():
                                class_key = int(class_label)
                            elif isinstance(class_label, (int, float)):
                                class_key = class_label
                            else:
                                class_key = float(class_label) if str(class_label).replace('.', '').isdigit() else class_label
                        except (ValueError, AttributeError):
                            class_key = class_label
                        
                        # Set min value (take minimum if exists, otherwise set directly)
                        if class_key in splitter._min_per_class:
                            splitter._min_per_class[class_key] = min(
                                splitter._min_per_class[class_key], min_val
                            )
                        else:
                            splitter._min_per_class[class_key] = min_val
                
                if 'max_per_class' in gaussian_data and hasattr(splitter, '_max_per_class'):
                    for class_label, max_val in gaussian_data['max_per_class'].items():
                        # Handle both string and numeric class labels
                        try:
                            if isinstance(class_label, str) and class_label.isdigit():
                                class_key = int(class_label)
                            elif isinstance(class_label, (int, float)):
                                class_key = class_label
                            else:
                                class_key = float(class_label) if str(class_label).replace('.', '').isdigit() else class_label
                        except (ValueError, AttributeError):
                            class_key = class_label
                        
                        # Set max value (take maximum if exists, otherwise set directly)
                        if class_key in splitter._max_per_class:
                            splitter._max_per_class[class_key] = max(
                                splitter._max_per_class[class_key], max_val
                            )
                        else:
                            splitter._max_per_class[class_key] = max_val
            
            # Update Nominal splitter data
            elif 'nominal_data' in splitter_update:
                nominal_data = splitter_update['nominal_data']
                print(f"         🏷️ Updating Nominal splitter...")
                
                # Update class distributions for nominal
                if 'class_distributions' in nominal_data:
                    if not hasattr(splitter, '_att_dist_per_class'):
                        print(f"         ⚠️ Splitter has no _att_dist_per_class")
                        continue
                    
                    for class_label, category_counts in nominal_data['class_distributions'].items():
                        # Handle both string and numeric class labels
                        try:
                            if isinstance(class_label, str) and class_label.isdigit():
                                class_key = int(class_label)
                            elif isinstance(class_label, (int, float)):
                                class_key = class_label
                            else:
                                class_key = float(class_label) if str(class_label).replace('.', '').isdigit() else class_label
                        except (ValueError, AttributeError):
                            class_key = class_label
                        
                        if class_key not in splitter._att_dist_per_class:
                            splitter._att_dist_per_class[class_key] = {}
                        
                        # Update category counts
                        for category, count in category_counts.items():
                            if category in splitter._att_dist_per_class[class_key]:
                                splitter._att_dist_per_class[class_key][category] += count
                            else:
                                splitter._att_dist_per_class[class_key][category] = count
                        
                        print(f"         ✅ Updated class {class_key} nominal distribution")
                
                # Update unique values set
                if 'unique_values' in nominal_data and hasattr(splitter, '_att_values'):
                    for value in nominal_data['unique_values']:
                        splitter._att_values.add(value)
                
                # Update total weight
                if 'total_weight' in nominal_data and hasattr(splitter, '_total_weight_observed'):
                    splitter._total_weight_observed += nominal_data['total_weight']
        
        print(f"      ✅ Splitter data update completed")
        return True
    
    def _apply_naive_bayes_update(self, node, update_data):
        """Apply Naive Bayes specific updates (correctness weights, etc.)."""
        print(f"   🧠 Applying Naive Bayes update...")
        
        # Update Naive Bayes correctness weights
        if 'mc_correct_weight' in update_data:
            if hasattr(node, '_mc_correct_weight'):
                node._mc_correct_weight = update_data['mc_correct_weight']
            else:
                node._mc_correct_weight = update_data['mc_correct_weight']
            print(f"      📊 Updated MC correct weight: {getattr(node, '_mc_correct_weight', 0)}")
        
        if 'nb_correct_weight' in update_data:
            if hasattr(node, '_nb_correct_weight'):
                node._nb_correct_weight = update_data['nb_correct_weight']
            else:
                node._nb_correct_weight = update_data['nb_correct_weight']
            print(f"      📊 Updated NB correct weight: {getattr(node, '_nb_correct_weight', 0)}")
        
        return True
    
    def _apply_complete_node_update(self, node, update_data):
        """Apply a complete node state update (all data at once)."""
        print(f"   🔄 Applying complete node update...")
        
        # Apply all update types in sequence
        success = True
        
        if 'leaf_stats' in update_data:
            success &= self._apply_leaf_stats_update(node, update_data['leaf_stats'])
        
        if 'splitter_data' in update_data:
            success &= self._apply_splitter_data_update(node, update_data['splitter_data'])
        
        if 'naive_bayes_data' in update_data:
            success &= self._apply_naive_bayes_update(node, update_data['naive_bayes_data'])
        
        print(f"   {'✅' if success else '❌'} Complete node update {'completed' if success else 'failed'}")
        return success
    
    def _apply_incremental_stats_update(self, node, update_data):
        """Apply incremental statistics updates (like single instance learning)."""
        print(f"   📈 Applying incremental stats update...")
        
        # This simulates learning from a single instance received from distributed system
        if 'instance' in update_data and 'class_label' in update_data:
            x = update_data['instance']
            y = update_data['class_label']
            w = update_data.get('weight', 1.0)
            
            print(f"      🎯 Learning from distributed instance: class={y}, weight={w}")
            
            # Apply the learning directly to the node
            node.learn_one(x, y, w=w, tree=self)
            
            print(f"      ✅ Incremental learning applied")
            return True
        
        return False
    
    def create_update_payload(self, node_id, update_type='complete_node'):
        """Create an update payload for a specific node that can be sent to other distributed processes.
        
        This method extracts the current state of a node and packages it for distribution.
        
        Parameters
        ----------
        node_id : int
            The ID of the node to create payload for
        update_type : str
            Type of update payload to create
        
        Returns
        -------
        dict or None
            Update payload dictionary, or None if node not found
        """
        node = self.get_node_by_id(node_id)
        if node is None:
            print(f"❌ Node {node_id} not found for payload creation")
            return None
        
        print(f"📦 CREATING UPDATE PAYLOAD:")
        print(f"   Node ID: {node_id} ({type(node).__name__})")
        print(f"   Payload type: {update_type}")
        
        payload = {
            'node_id': node_id,
            'update_type': update_type,
            'timestamp': __import__('time').time(),
            'source_tree_id': id(self),
            'data': {}
        }
        
        if update_type in ['complete_node', 'leaf_stats']:
            # Add leaf statistics
            payload['data']['leaf_stats'] = {
                'stats': dict(getattr(node, 'stats', {})),
                'total_weight': getattr(node, 'total_weight', 0),
            }
        
        if update_type in ['complete_node', 'splitter_data']:
            # Add splitter data
            payload['data']['splitter_data'] = {
                'splitters': self._extract_splitter_data_for_callback(node)
            }
        
        if update_type in ['complete_node', 'naive_bayes_data']:
            # Add Naive Bayes data
            payload['data']['naive_bayes_data'] = {
                'mc_correct_weight': getattr(node, '_mc_correct_weight', 0),
                'nb_correct_weight': getattr(node, '_nb_correct_weight', 0),
            }
        
        print(f"   ✅ Payload created with {len(payload['data'])} data sections")
        return payload
    
    def apply_split_event(self, split_data):
        """Apply a split event to reconstruct tree structure from distributed training.
        
        This method allows an inference process to reconstruct the tree structure
        by applying split events received from a distributed training process.
        
        Parameters
        ----------
        split_data : dict
            Dictionary containing split event information with the following structure:
            {
                'original_leaf_id': int,
                'split_node': {
                    'node_id': int,
                    'node_type': str,
                    'branch_params': dict,
                    'stats': dict,
                    'depth': int
                },
                'new_leaves': [
                    {
                        'node_id': int,
                        'stats': dict,
                        'depth': int
                    },
                    ...
                ]
            }
        
        Returns
        -------
        bool
            True if split event was successfully applied, False otherwise
        """
        print("=" * 70)
        print("📥 APPLYING SPLIT EVENT")
        print("=" * 70)
        
        original_leaf_id = split_data['original_leaf_id']
        split_node_info = split_data['split_node']
        new_leaves_info = split_data['new_leaves']
        
        print(f"Original leaf ID: {original_leaf_id}")
        print(f"Split node ID: {split_node_info['node_id']}")
        print(f"Split node type: {split_node_info['node_type']}")
        print(f"Branch params: {split_node_info['branch_params']}")
        print(f"New leaves: {[leaf['node_id'] for leaf in new_leaves_info]}")
        print(f"   📊 Split data: {split_data}")
        print()
        
        try:
            # Create new leaf children FIRST
            new_leaves = []
            for leaf_info in new_leaves_info:
                leaf = self._create_leaf_from_info(leaf_info)
                new_leaves.append(leaf)
                
            # Create the split node with children
            # Note: The children are attached during construction (left/right params)
            split_node = self._create_split_node(split_node_info, new_leaves)
            
            # Verify children are attached (for debugging)
            print(f"   Split node has {len(split_node.children)} children:")
            for i, child in enumerate(split_node.children):
                child_id = new_leaves_info[i]['node_id']
                print(f"      Child {i}: {type(child).__name__} (will be node_id={child_id})")
            
            # Replace the original leaf with the new split node
            if self._root is None or original_leaf_id == 0:
                # Root split case
                self._root = split_node
                print(f"✅ Replaced root with split node ID={split_node_info['node_id']}")
            else:
                # Non-root split: find parent and replace child
                parent_node, child_index = self._find_parent_and_index(original_leaf_id)
                
                if parent_node is not None:
                    # Replace the child at the found index
                    parent_node.children[child_index] = split_node
                    print(f"✅ Replaced child at index {child_index} of parent node ID={getattr(parent_node, 'node_id', 'unknown')}")
                    print(f"   Original leaf ID: {original_leaf_id} → New split node ID: {split_node_info['node_id']}")
                else:
                    print(f"⚠️  Warning: Could not find parent for leaf ID {original_leaf_id}")
                    print(f"   This may indicate the tree structure is inconsistent")
            
            # Remove the original leaf from registry (it's being replaced)
            if original_leaf_id in self._node_registry:
                old_leaf = self._node_registry.pop(original_leaf_id)
                print(f"   🗑️  Removed original leaf ID={original_leaf_id} from registry")
            
            # Register new nodes in the model's node registry
            # This assigns node_id attributes and adds to registry for O(1) lookup
            self._register_node(split_node, split_node_info['node_id'])
            for leaf, leaf_info in zip(new_leaves, new_leaves_info):
                self._register_node(leaf, leaf_info['node_id'])
            print(f"   Registry now contains: {list(self._node_registry.keys())}")
            
            print(f"✅ Split event applied successfully")
            print(f"   Model state: {self.n_nodes} nodes, height {self.height}")
            print()

            # Check all nodes
            for node_id, node in self._node_registry.items():
                print(f"   Node ID: {node_id}, Type: {type(node).__name__}")
                if hasattr(node, 'stats'):
                    print(f"      Stats: {node.stats}")
                    print(f"      Total weight: {node.total_weight}")
                    if hasattr(node, 'depth'):
                        print(f"      Depth: {node.depth}")
                if hasattr(node, 'feature'):
                    print(f"      Feature: {node.feature}")
                if hasattr(node, 'threshold'):
                    print(f"      Threshold: {node.threshold}")
                if hasattr(node, 'value'):
                    print(f"      Value: {node.value}")
                if hasattr(node, 'children'):
                    print(f"      Children count: {len(node.children)}")
                print()

                if hasattr(node, 'splitters') and node.splitters is not None:
                    for feat, splitter in node.splitters.items():
                        print(f"      Splitter for feature '{feat}': {type(splitter).__name__}")
                        if isinstance(splitter, GaussianSplitter):
                            print(f"         Distributions: {splitter._att_dist_per_class}")
                            print(f"         Min per class: {splitter._min_per_class}")
                            print(f"         Max per class: {splitter._max_per_class}")
                    print()
            
            return True
            
        except Exception as e:
            print(f"❌ Error applying split event: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_leaf_from_info(self, leaf_info):
        """Create a leaf node from leaf information.
        
        Parameters
        ----------
        leaf_info : dict
            Dictionary containing leaf information:
            {
                'node_id': int,
                'stats': dict,
                'depth': int
            }
        
        Returns
        -------
        HTLeaf
            Created leaf node (type depends on leaf_prediction setting)
        """
        stats = {int(k): v for k, v in leaf_info['stats'].items()}
        depth = leaf_info.get('depth', 0)
        
        print(f"   Creating leaf from info: {leaf_info}")
        
        # Create leaf node based on leaf_prediction setting
        # CRITICAL: Use self.splitter template, NOT None!
        # This matches what _new_leaf() does (line 411)
        if self._leaf_prediction == self._MAJORITY_CLASS:
            leaf = LeafMajorityClass(stats=stats, depth=depth, splitter=self.splitter)
        elif self._leaf_prediction == self._NAIVE_BAYES:
            leaf = LeafNaiveBayes(stats=stats, depth=depth, splitter=self.splitter)
        else:  # Naive Bayes Adaptive (default)
            leaf = LeafNaiveBayesAdaptive(stats=stats, depth=depth, splitter=self.splitter)

        return leaf
    
    def _create_split_node(self, split_node_info, children):
        """Create a split node from split event data with children.
        
        Parameters
        ----------
        split_node_info : dict
            Dictionary containing split node information:
            {
                'node_id': int,
                'node_type': str,
                'branch_params': dict,
                'stats': dict,
                'depth': int
            }
        children : list
            List of child nodes (leaves or branches)
        
        Returns
        -------
        DTBranch
            Created branch node (type depends on node_type)
        """
        from .nodes.branch import NumericBinaryBranch, NominalBinaryBranch
        from .nodes.branch import NumericMultiwayBranch, NominalMultiwayBranch
        
        node_type = split_node_info['node_type']
        branch_params = split_node_info['branch_params']
        stats = split_node_info.get('stats', {})
        depth = split_node_info.get('depth', 0)
        
        # Convert stats from string keys to int
        stats_dict = {int(k): v for k, v in stats.items()} if stats else {}
        
        if node_type == 'NumericBinaryBranch':
            node = NumericBinaryBranch(
                stats=stats_dict,
                feature=branch_params['feature'],
                threshold=branch_params['threshold'],
                depth=depth,
                left=children[0],
                right=children[1]
            )
        elif node_type == 'NominalBinaryBranch':
            node = NominalBinaryBranch(
                stats=stats_dict,
                feature=branch_params['feature'],
                value=branch_params['value'],
                depth=depth,
                left=children[0],
                right=children[1]
            )
        elif node_type == 'NumericMultiwayBranch':
            node = NumericMultiwayBranch(
                stats=stats_dict,
                feature=branch_params['feature'],
                depth=depth,
                *children  # Multiway branches accept variable children
            )
        elif node_type == 'NominalMultiwayBranch':
            node = NominalMultiwayBranch(
                stats=stats_dict,
                feature=branch_params['feature'],
                depth=depth,
                *children
            )
        else:
            raise ValueError(f"Unknown split node type: {node_type}")
        
        return node
    
    def _find_parent_and_index(self, target_node_id):
        """Find the parent node and child index for a given node ID.
        
        This method traverses the tree to find which parent node contains
        the target node as a child, and at which index.
        
        Parameters
        ----------
        target_node_id : int
            The ID of the node to find the parent for
        
        Returns
        -------
        tuple (DTBranch, int) or (None, None)
            The parent node and the index of the target node in parent's children,
            or (None, None) if not found
        """
        if self._root is None:
            return None, None
        
        # If target is root, it has no parent
        if hasattr(self._root, 'node_id') and self._root.node_id == target_node_id:
            return None, None
        
        # BFS traversal to find the parent
        from collections import deque
        queue = deque([self._root])
        
        while queue:
            current = queue.popleft()
            
            # Check if this node is a branch (has children)
            if hasattr(current, 'children') and current.children:
                # Check each child
                for idx, child in enumerate(current.children):
                    # Found it!
                    if hasattr(child, 'node_id') and child.node_id == target_node_id:
                        return current, idx
                    
                    # Add child to queue for further traversal
                    if hasattr(child, 'children'):
                        queue.append(child)
        
        # Not found
        return None, None
