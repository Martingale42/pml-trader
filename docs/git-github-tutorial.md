# Git and GitHub Tutorial for Team Development

This tutorial will guide you through using Git and GitHub for version control and team collaboration on Python projects. We'll cover everything from basic concepts to practical workflows, with detailed examples and common scenarios.

## Table of Contents
1. Basic Concepts
2. Initial Setup
3. Working with Branches
4. Managing Changes 
5. Remote Repository Operations
6. Team Collaboration
7. Best Practices
8. Common Scenarios and Solutions

## 1. Basic Concepts

Git is a distributed version control system that helps you track changes in your code. Think of it like a time machine for your project that lets you:
- Save snapshots of your code at different points in time
- Create alternate versions to experiment with new features
- Collaborate with others without overwriting each other's work

### Key Terms Explained

#### Repository (repo)
Think of a repository as a project folder with a special `.git` directory that tracks all changes. It's like having a detailed history book of your project, recording who changed what and when.

#### Commit
A commit is like taking a snapshot of your code. Each commit has:
- A unique identifier (hash)
- A message describing what changed
- Information about who made the change and when
- A reference to the previous commit(s)

#### Branch
Imagine branches as parallel universes of your code. Each branch can evolve independently without affecting others. They're perfect for:
- Developing new features without risking the stable code
- Experimenting with different solutions
- Working on multiple features simultaneously

#### Remote
A remote is a copy of your repository hosted on a server (like GitHub). Think of it as a central hub where team members can share their changes and collaborate.

## 2. Initial Setup

### Creating a New Repository

Let's set up a new project with proper Git configuration:

```bash
# Create project directory
mkdir my_project
cd my_project

# Initialize Git repository
git init

# Create comprehensive .gitignore
cat > .gitignore << EOL
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
.env
.venv
env/
venv/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Jupyter Notebook
.ipynb_checkpoints

# Project specific
data/raw/
data/processed/
logs/
*.log
EOL

# Create initial project structure
mkdir -p src/models src/utils tests docs

# Create basic project files
touch README.md
touch src/__init__.py
touch src/models/__init__.py
touch src/utils/__init__.py

# Add all files to staging
git add .

# Create initial commit
git commit -m "Initial commit: Set up project structure

- Create basic directory structure
- Add .gitignore file
- Add empty package files"

# Rename default branch to main (current best practice)
git branch -M main
```

### Connecting to GitHub

After creating a repository on GitHub, connect your local repository:

```bash
# Add remote repository
git remote add origin https://github.com/USERNAME/REPOSITORY.git

# Verify remote was added correctly
git remote -v

# Push code to GitHub and set up tracking
git push -u origin main
```

## 3. Working with Branches

### Branch Management with Examples

Let's walk through a typical feature development workflow:

```bash
# Start by making sure you're up to date
git checkout main
git pull origin main

# Create a new feature branch
git checkout -b feature/user-authentication

# Make some changes
echo "def authenticate_user(username, password):" > src/utils/auth.py
git add src/utils/auth.py
git commit -m "feat(auth): Add basic user authentication function"

# Make more changes
echo "def hash_password(password):" >> src/utils/auth.py
git add src/utils/auth.py
git commit -m "feat(auth): Add password hashing function"

# Push your feature branch to remote
git push -u origin feature/user-authentication
```

### Visual Branch Structure

Here's how your branches might look:

```
main       o---o---o---o
                     \
feature            o---o---o
```

### Branch Naming Conventions

Use descriptive prefixes for different types of branches:
- `feature/add-login`
- `bugfix/fix-memory-leak`
- `hotfix/critical-security-patch`
- `release/v1.0.0`

## 4. Managing Changes

### Real-world Development Example

Let's walk through developing a new feature:

```bash
# Start a new feature
git checkout -b feature/hmm-implementation

# Create new file
cat > src/models/hmm.py << EOL
class HiddenMarkovModel:
    def __init__(self, n_states):
        self.n_states = n_states
        self.transitions = None
        self.emissions = None
EOL

# Stage and commit
git add src/models/hmm.py
git commit -m "feat(hmm): Add HMM class skeleton

- Add basic class structure
- Define constructor with n_states parameter"

# Add more functionality
cat >> src/models/hmm.py << EOL

    def fit(self, observations):
        # Implementation coming soon
        pass
EOL

git add src/models/hmm.py
git commit -m "feat(hmm): Add fit method placeholder

TODO: Implement forward-backward algorithm"
```

### Detailed Commit Message Examples

Good commit messages tell a story. Here are some examples:

```
feat(hmm): Implement forward-backward algorithm

- Add forward pass calculation
- Add backward pass calculation
- Add probability matrix normalization
- Add convergence checking
- Add unit tests for all new functionality

Performance: O(N*T) where N is number of states and T is sequence length
```

```
fix(memory): Resolve memory leak in HMM training

The transition matrix was being recreated unnecessarily in each
iteration, causing memory usage to grow linearly with training time.

- Move matrix initialization outside the training loop
- Add memory usage tracking in debug mode
- Add regression test to verify fix

Fixes #123
```

## 5. Remote Repository Operations

### Synchronization Patterns

Here's how to handle common synchronization scenarios:

```bash
# Scenario 1: Get updates without changing your work
git fetch origin
git status  # See how many commits you're behind
git log origin/main  # Review new commits

# Scenario 2: Update your branch with main's changes
git checkout feature/hmm-implementation
git fetch origin
git merge origin/main  # Or use rebase for cleaner history

# Scenario 3: Multiple remotes (e.g., upstream and fork)
git remote add upstream https://github.com/original/repo.git
git fetch upstream
git merge upstream/main
```

### Resolving Merge Conflicts

When you encounter a merge conflict:

```bash
# Start merge
git merge develop

# You get a conflict in hmm.py
# The file will look like:
<<<<<<< HEAD
def fit(self, observations):
    # Your implementation
=======
def fit(self, observations, n_iterations=100):
    # Their implementation
>>>>>>> develop

# Edit the file to resolve conflict
# Then:
git add src/models/hmm.py
git commit -m "merge: Resolve HMM fit method conflict"
```

## 6. Team Collaboration

### Pull Request Workflow Example

1. Prepare your feature branch:
```bash
# Update your branch with latest main
git checkout feature/hmm-implementation
git fetch origin
git merge origin/main

# Review your changes
git log --oneline main..HEAD
```

2. Push to GitHub:
```bash
git push origin feature/hmm-implementation
```

3. Create Pull Request on GitHub with:
```markdown
# Pull Request: Add Hidden Markov Model Implementation

## Changes
- Add HMM class with forward-backward algorithm
- Add model training functionality
- Add prediction methods
- Add comprehensive unit tests

## Testing
- All unit tests passing
- Tested with sample data
- Memory usage monitored and stable

## Notes
- Requires numpy >= 1.20
- Performance optimized for sparse transition matrices
```

## 7. Best Practices

### Code Review Checklist

When reviewing Pull Requests:

1. Code Quality
   - Does the code follow project style guidelines?
   - Is the code well-documented?
   - Are there appropriate unit tests?

2. Git Hygiene
   - Are commits atomic and well-described?
   - Is the branch up to date with main?
   - Are there any merge conflicts?

3. Functionality
   - Does the code do what it claims?
   - Are edge cases handled?
   - Is error handling appropriate?

### Branch Protection Rules

Set up these rules in GitHub:
```
main branch:
- Require pull request reviews
- Require status checks to pass
- Require linear history
- Include administrators
```

## 8. Common Scenarios and Solutions

### Scenario 1: Oops, committed to wrong branch

```bash
# Save your changes
git log -1  # Copy the commit hash
git checkout correct-branch
git cherry-pick <commit-hash>
git checkout previous-branch
git reset --hard HEAD~1
```

### Scenario 2: Need to undo last commit

```bash
# Undo commit but keep changes staged
git reset --soft HEAD~1

# Undo commit and unstage changes
git reset HEAD~1

# Completely remove last commit and changes
git reset --hard HEAD~1
```

### Scenario 3: Accidentally pushed sensitive data

```bash
# Remove sensitive file from git history
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch config.secret" \
  --prune-empty --tag-name-filter cat -- --all

# Force push the changes
git push origin --force --all
```

Remember: Git is powerful but forgiving. Most mistakes can be undone with the right commands. When in doubt, create a backup branch before trying complex operations:

```bash
# Create backup branch
git branch backup-before-operation

# If things go wrong
git checkout backup-before-operation
```

These detailed examples and scenarios should help you better understand how to use Git and GitHub effectively in your development workflow.