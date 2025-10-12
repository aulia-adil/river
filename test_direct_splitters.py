"""
Direct test of splitter objects to see their internal structure.
This bypasses the full River import to focus on the splitter classes.
"""

import sys
import os
sys.path.append('/root/river')

def test_splitter_structure():
    print("🔍 DIRECT SPLITTER OBJECT INSPECTION")
    print("=" * 50)
    
    # Let's create splitter objects directly and inspect them
    try:
        # Try to import splitter classes directly
        from river.tree.splitter import GaussianSplitter, NominalSplitterClassif
        
        print("📊 GAUSSIAN SPLITTER (for numeric features like age):")
        print("-" * 30)
        
        # Create a Gaussian splitter
        gaussian_splitter = GaussianSplitter()
        
        # Simulate some training data
        # Class 0: younger people (ages 20-30)
        for age in [22, 25, 28, 24, 26]:
            gaussian_splitter.update(age, 0, 1.0)
            
        # Class 1: older people (ages 40-50) 
        for age in [42, 45, 48, 44, 46]:
            gaussian_splitter.update(age, 1, 1.0)
        
        print("Gaussian Splitter attributes:")
        for attr in dir(gaussian_splitter):
            if not attr.startswith('__'):
                try:
                    value = getattr(gaussian_splitter, attr)
                    if not callable(value):
                        print(f"  {attr}: {value}")
                except:
                    pass
        
        print(f"\n📈 Testing conditional probability:")
        print(f"P(age=30|class=0): {gaussian_splitter.cond_proba(30, 0)}")
        print(f"P(age=30|class=1): {gaussian_splitter.cond_proba(30, 1)}")
        print(f"P(age=45|class=0): {gaussian_splitter.cond_proba(45, 0)}")
        print(f"P(age=45|class=1): {gaussian_splitter.cond_proba(45, 1)}")
        
        print("\n" + "="*50)
        print("📊 NOMINAL SPLITTER (for categorical features like education):")
        print("-" * 30)
        
        # Create a Nominal splitter
        nominal_splitter = NominalSplitterClassif()
        
        # Simulate some training data
        # Class 0: mostly high school
        for education in ['high_school', 'high_school', 'college', 'high_school']:
            nominal_splitter.update(education, 0, 1.0)
            
        # Class 1: mostly college
        for education in ['college', 'college', 'graduate', 'college', 'graduate']:
            nominal_splitter.update(education, 1, 1.0)
        
        print("Nominal Splitter attributes:")
        for attr in dir(nominal_splitter):
            if not attr.startswith('__'):
                try:
                    value = getattr(nominal_splitter, attr)
                    if not callable(value):
                        print(f"  {attr}: {value}")
                except:
                    pass
        
        print(f"\n📈 Testing conditional probability:")
        print(f"P(education='high_school'|class=0): {nominal_splitter.cond_proba('high_school', 0)}")
        print(f"P(education='high_school'|class=1): {nominal_splitter.cond_proba('high_school', 1)}")
        print(f"P(education='college'|class=0): {nominal_splitter.cond_proba('college', 0)}")
        print(f"P(education='college'|class=1): {nominal_splitter.cond_proba('college', 1)}")
        
    except ImportError as e:
        print(f"❌ Cannot import splitter classes: {e}")
        print("Let's try to find the splitter files...")
        
        # Look for splitter files
        import glob
        splitter_files = glob.glob('/root/river/river/tree/*splitter*')
        print(f"Found splitter files: {splitter_files}")

if __name__ == "__main__":
    test_splitter_structure()