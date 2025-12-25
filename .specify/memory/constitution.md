# Constitution: Physical AI & Humanoid Robotics: From ROS 2 to Vision-Language-Action

## Project Mission
This educational book serves as a comprehensive guide to humanoid robotics, bridging traditional robotics middleware with cutting-edge AI integration. It shall provide technically accurate, pedagogically sound content that enables intermediate-to-advanced learners to build, simulate, and control humanoid robots using modern tools and best practices.

## 1. Content Quality & Accuracy

### Technical Accuracy Requirements
- All technical information MUST be verified against official documentation from ROS 2, NVIDIA Isaac, Gazebo, Unity Robotics, OpenAI, and other authoritative sources
- Zero tolerance for hallucinated APIs, deprecated features, or incorrect code examples
- All technical claims MUST be verifiable through official documentation or reproducible examples
- Prefer 2024-2025 era conventions and best practices over legacy approaches
- Deprecated or legacy approaches MUST be explicitly labeled and explained with migration paths

### Fact-Checking Protocol
- All code examples MUST be validated against the latest stable versions (ROS 2 Humble/Iron/Jazzy, Python 3.10+, NVIDIA Isaac Sim 2024.x+)
- Technical claims about API behavior MUST be confirmed through official documentation
- Performance claims MUST be accompanied by benchmarks or references to authoritative sources
- When uncertain about technical details, explicitly state "consult official documentation" and provide specific links

## 2. Educational Style & Pedagogy

### Progressive Learning Structure
- Assume intermediate Python/ROS knowledge but explain advanced humanoid concepts in depth
- Use clear, concise language with short paragraphs (3-5 sentences maximum)
- Structure content in progressive difficulty: foundational concepts → intermediate applications → advanced integrations
- Each section MUST include: why this matters, real-world applications, common pitfalls, and troubleshooting tips

### Tone and Approach
- Maintain "explain like I'm a motivated robotics engineer" tone — professional yet enthusiastic
- Use analogies and real-world examples to explain complex concepts
- Include historical context where relevant to understand current best practices
- Acknowledge complexity while making concepts accessible through clear explanations

### Learning Objectives
- Each chapter MUST have clear learning objectives stated at the beginning
- Include hands-on exercises and practical examples in every major section
- Provide "quick wins" early to maintain engagement while building toward complex concepts
- End each module with key takeaways and suggested further reading

## 3. Code Snippets & Examples

### Code Quality Standards
- All code examples MUST be valid, runnable (or very close) ROS 2/Python syntax
- Use modern rclpy, Python 3.10+, ROS 2 Humble/Iron/Jazzy conventions consistently
- Always include necessary imports, context setup, and minimal runnable examples
- Use proper indentation and follow PEP 8 style guidelines
- Comment important lines to explain non-obvious behavior or critical parameters

### Code Structure Requirements
- Prefer ROS 2 services/actions over legacy ROS 1 patterns
- Use composition patterns and lifecycle nodes where appropriate
- Include error handling and logging best practices in examples
- Provide both minimal working examples and more complete implementations
- Include timing considerations and real-time constraints where relevant

### Testing and Validation
- All code examples SHOULD be tested in simulation environments before inclusion
- Include expected output or behavior where applicable
- Provide troubleshooting tips for common runtime errors
- Flag experimental or unstable APIs clearly

## 4. Visual & Structural Standards

### Diagram Requirements
- Use Mermaid diagrams for every architecture, data-flow, or system interaction
- Include URDF tree diagrams, Nav2 behavior trees, and system integration diagrams
- Provide PlantUML for sequence diagrams showing message flow
- Every diagram MUST have a clear title and explanatory caption

### Visual Content Standards
- Include high-quality screenshots showing Isaac Sim, Gazebo, Unity environments
- Screenshots MUST clearly show the relevant UI elements or simulation state
- Use consistent color schemes and visual styling throughout
- All images MUST have descriptive alt text for accessibility

### Document Structure
- Use consistent Docusaurus Markdown/MDX structure: H1 for module title, H2/H3 for sections
- Use admonitions (note, tip, warning, danger) appropriately for emphasis
- Include cross-references between related concepts across modules
- Maintain consistent terminology throughout the book

## 5. Module-Specific Guidelines

### Module 1: ROS 2 Foundations
- Emphasize real-time distributed systems concepts and ROS 2 architecture
- Focus on rclpy best practices for performance and reliability
- Cover humanoid-specific URDF complexity: joints, transmissions, sensors, and multi-body dynamics
- Include real-time constraints and QoS profile considerations

### Module 2: Simulation Environments (Gazebo & Unity)
- Address physics realism vs. rendering trade-offs explicitly
- Cover sensor noise simulation and realistic sensor modeling
- Focus on Unity for HRI (Human-Robot Interaction) visualization
- Include performance optimization strategies for complex humanoid models

### Module 3: NVIDIA Isaac Ecosystem
- Emphasize photorealism and synthetic data generation for training
- Cover hardware-accelerated ROS nodes and performance optimization
- Address bipedal locomotion challenges and control strategies
- Include integration with NVIDIA Isaac ROS and cuRobo for motion planning

### Module 4: Vision-Language-Action Integration & Capstone
- Show complete LLM chaining: Whisper → LLM planner → ROS 2 actions
- Address safety guardrails and validation mechanisms
- Cover simulation-to-real transfer challenges and solutions
- Include comprehensive capstone project integrating all previous modules

## 6. Book-wide Consistency & Maintainability

### Terminology Standards
- Use consistent terminology: "humanoid robot", "bipedal locomotion", "end-effector", "manipulator"
- Define domain-specific terms when first introduced
- Use technical precision while maintaining accessibility
- Cross-reference concepts between modules where knowledge builds

### Versioning and Dependencies
- Prefer latest stable versions as of December 2025
- Include version compatibility matrices where applicable
- Note breaking changes and migration paths clearly
- Maintain compatibility with LTS ROS 2 distributions

### Licensing and Distribution
- Apply CC-BY 4.0 license for educational reuse and modification
- Attribute all third-party code examples and resources appropriately
- Include proper citations for research papers and technical documentation
- Maintain Git-friendly formatting with short lines and atomic commits

### Maintenance Considerations
- Structure content for easy updates as technologies evolve
- Include deprecation warnings for experimental features
- Plan for regular content updates aligned with new ROS 2 and Isaac releases
- Maintain clear contribution guidelines for community involvement

## 7. Governance & AI Agent Rules

### Constitution Priority
- This constitution serves as the highest-priority context for all content generation
- All generated content MUST be validated against these principles before acceptance
- When conflicts arise, consult this constitution first, then seek clarification

### Technical Accuracy Protocol
- If uncertain about technical details, explicitly state uncertainty and recommend official documentation
- Prevent over-simplification that loses technical accuracy
- Prevent over-complexity that confuses readers without pedagogical value

### Content Integration Requirements
- Every generated chapter MUST be self-contained while linking to prerequisite knowledge
- Include clear prerequisites and learning path dependencies
- Maintain consistent learning objectives across all modules
- Ensure smooth transitions between modules and concepts

### Quality Assurance
- All content MUST undergo technical review against official documentation
- Code examples MUST be validated in simulation environments
- Visual content MUST enhance understanding rather than decorate
- Educational effectiveness MUST be validated through learning objectives achievement

## 8. Acceptance Criteria

### Content Standards
- [ ] All technical claims verified against official documentation
- [ ] Code examples tested and validated
- [ ] Learning objectives clearly stated and met
- [ ] Visual content enhances understanding
- [ ] Cross-references maintain consistency
- [ ] Terminology used consistently throughout

### Educational Standards
- [ ] Appropriate difficulty progression maintained
- [ ] Pedagogical approach consistent with target audience
- [ ] Practical examples included in every major section
- [ ] Troubleshooting and best practices clearly documented
- [ ] Real-world applications connected to theoretical concepts

This constitution shall serve as the foundational governance document for all content generation, ensuring consistency, accuracy, and educational effectiveness throughout the book project.