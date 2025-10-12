"""
Simple demonstration of what's inside the 'obs' (splitter) objects.
This shows the exact data structures you need to serialize for Kafka.
"""

class MockGaussian:
    """Simple mock of the Gaussian distribution class."""
    def __init__(self):
        self.n_samples = 0
        self._sum = 0.0
        self._sum_squares = 0.0
        
    def update(self, x, w=1.0):
        self.n_samples += w
        self._sum += x * w
        self._sum_squares += (x * x) * w
    
    @property
    def mean(self):
        return self._sum / self.n_samples if self.n_samples > 0 else 0.0
    
    @property
    def variance(self):
        if self.n_samples <= 1:
            return 1.0
        mean = self.mean
        return (self._sum_squares / self.n_samples) - (mean * mean)
    
    def __call__(self, x):
        """Calculate probability density."""
        import math
        variance = self.variance
        if variance <= 0:
            return 1.0
        mean = self.mean
        return (1.0 / math.sqrt(2 * math.pi * variance)) * math.exp(-0.5 * ((x - mean) ** 2) / variance)

class MockGaussianSplitter:
    """Mock version of GaussianSplitter to show the data structure."""
    
    def __init__(self):
        self._min_per_class = {}
        self._max_per_class = {}
        self._att_dist_per_class = {}
        
    def update(self, att_val, target_val, w=1.0):
        if att_val is None:
            return
            
        try:
            val_dist = self._att_dist_per_class[target_val]
            if att_val < self._min_per_class[target_val]:
                self._min_per_class[target_val] = att_val
            if att_val > self._max_per_class[target_val]:
                self._max_per_class[target_val] = att_val
        except KeyError:
            val_dist = MockGaussian()
            self._att_dist_per_class[target_val] = val_dist
            self._min_per_class[target_val] = att_val
            self._max_per_class[target_val] = att_val
            
        val_dist.update(att_val, w)
    
    def cond_proba(self, att_val, target_val):
        if target_val in self._att_dist_per_class:
            obs = self._att_dist_per_class[target_val]
            return obs(att_val)
        else:
            return 0.0

class MockNominalSplitter:
    """Mock version of NominalSplitterClassif to show the data structure."""
    
    def __init__(self):
        self._total_weight_observed = 0.0
        self._missing_weight_observed = 0.0
        self._att_dist_per_class = {}
        self._att_values = set()
    
    def update(self, att_val, target_val, w=1.0):
        if att_val is None:
            self._missing_weight_observed += w
        else:
            self._att_values.add(att_val)
            
            if target_val not in self._att_dist_per_class:
                self._att_dist_per_class[target_val] = {}
            
            if att_val not in self._att_dist_per_class[target_val]:
                self._att_dist_per_class[target_val][att_val] = 0.0
                
            self._att_dist_per_class[target_val][att_val] += w
        
        self._total_weight_observed += w
    
    def cond_proba(self, att_val, target_val):
        if target_val not in self._att_dist_per_class:
            return 0.0
            
        class_dist = self._att_dist_per_class[target_val]
        
        if att_val not in class_dist:
            return 0.0
        
        value = class_dist[att_val]
        total = sum(class_dist.values())
        return value / total if total > 0 else 0.0

def demonstrate_splitter_internals():
    print("🔍 DETAILED VIEW OF SPLITTER (obs) OBJECTS")
    print("=" * 60)
    
    # 1. GAUSSIAN SPLITTER (for numeric features like age)
    print("\n📊 GAUSSIAN SPLITTER - Numeric Feature (age)")
    print("-" * 40)
    
    age_splitter = MockGaussianSplitter()
    
    # Train with age data
    print("Training with age data:")
    ages_class_0 = [22, 25, 28, 24, 26, 23, 27]  # Younger people
    ages_class_1 = [42, 45, 48, 44, 46, 43, 47]  # Older people
    
    for age in ages_class_0:
        age_splitter.update(age, 0, 1.0)
        print(f"  Added age={age} for class=0")
    
    for age in ages_class_1:
        age_splitter.update(age, 1, 1.0)
        print(f"  Added age={age} for class=1")
    
    print(f"\n🔍 INTERNAL DATA STRUCTURE:")
    print(f"  _min_per_class: {age_splitter._min_per_class}")
    print(f"  _max_per_class: {age_splitter._max_per_class}")
    print(f"  _att_dist_per_class keys: {list(age_splitter._att_dist_per_class.keys())}")
    
    for class_val, gaussian in age_splitter._att_dist_per_class.items():
        print(f"  Class {class_val} Gaussian:")
        print(f"    n_samples: {gaussian.n_samples}")
        print(f"    mean: {gaussian.mean:.2f}")
        print(f"    variance: {gaussian.variance:.2f}")
        print(f"    sum: {gaussian._sum}")
        print(f"    sum_squares: {gaussian._sum_squares}")
    
    print(f"\n📈 CONDITIONAL PROBABILITIES:")
    test_ages = [25, 35, 45]
    for age in test_ages:
        prob_0 = age_splitter.cond_proba(age, 0)
        prob_1 = age_splitter.cond_proba(age, 1)
        print(f"  P(age={age}|class=0) = {prob_0:.6f}")
        print(f"  P(age={age}|class=1) = {prob_1:.6f}")
    
    # 2. NOMINAL SPLITTER (for categorical features like education)
    print("\n\n📊 NOMINAL SPLITTER - Categorical Feature (education)")
    print("-" * 40)
    
    edu_splitter = MockNominalSplitter()
    
    # Train with education data
    print("Training with education data:")
    edu_class_0 = ['high_school', 'high_school', 'college', 'high_school', 'college']
    edu_class_1 = ['college', 'graduate', 'college', 'graduate', 'college', 'graduate']
    
    for edu in edu_class_0:
        edu_splitter.update(edu, 0, 1.0)
        print(f"  Added education={edu} for class=0")
    
    for edu in edu_class_1:
        edu_splitter.update(edu, 1, 1.0)
        print(f"  Added education={edu} for class=1")
    
    print(f"\n🔍 INTERNAL DATA STRUCTURE:")
    print(f"  _total_weight_observed: {edu_splitter._total_weight_observed}")
    print(f"  _missing_weight_observed: {edu_splitter._missing_weight_observed}")
    print(f"  _att_values (unique values): {edu_splitter._att_values}")
    print(f"  _att_dist_per_class: {edu_splitter._att_dist_per_class}")
    
    print(f"\n📊 DETAILED COUNTS:")
    for class_val, value_counts in edu_splitter._att_dist_per_class.items():
        print(f"  Class {class_val}:")
        for value, count in value_counts.items():
            print(f"    {value}: {count} occurrences")
    
    print(f"\n📈 CONDITIONAL PROBABILITIES:")
    test_values = ['high_school', 'college', 'graduate']
    for value in test_values:
        prob_0 = edu_splitter.cond_proba(value, 0)
        prob_1 = edu_splitter.cond_proba(value, 1)
        print(f"  P(education='{value}'|class=0) = {prob_0:.6f}")
        print(f"  P(education='{value}'|class=1) = {prob_1:.6f}")
    
    print(f"\n" + "="*60)
    print("💡 KEY INSIGHTS FOR YOUR KAFKA SERIALIZATION:")
    print("="*60)
    print("1. GAUSSIAN SPLITTERS contain:")
    print("   - Per-class statistics: min, max, mean, variance, sample count")
    print("   - Full Gaussian distribution parameters")
    print("   - Can calculate P(value|class) using probability density function")
    print()
    print("2. NOMINAL SPLITTERS contain:")
    print("   - Per-class-per-value frequency counts")
    print("   - Set of all observed values")
    print("   - Can calculate P(value|class) using frequency ratios")
    print()
    print("3. THIS IS WHAT GOES INTO YOUR KAFKA MESSAGES!")
    print("   - All this statistical data must be serialized")
    print("   - Inference processes reconstruct identical predictions")
    print("   - Each 'obs' object maintains complete feature models")

if __name__ == "__main__":
    demonstrate_splitter_internals()