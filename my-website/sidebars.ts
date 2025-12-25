import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  // By default, Docusaurus generates a sidebar from the docs folder structure
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Module 1: ROS 2 Foundations',
      items: [
        'module1/intro',
        'module1/nodes',
        'module1/topics',
        'module1/services-actions',
        'module1/python-agents-rclpy',
        'module1/urdf-humanoids',
        'module1/summary',
        'module1/review-checklist'
      ],
    },
    {
      type: 'category',
      label: 'Module 2: Simulation Environments',
      items: [
        'module2/intro',
        'module2/gazebo-fundamentals',
        'module2/physics-humanoids',
        'module2/sensor-simulation',
        'module2/unity-hri',
        'module2/optimization-best-practices',
        'module2/advanced-techniques',
        'module2/summary',
        'module2/review-checklist'
      ],
    },
    {
      type: 'category',
      label: 'Module 3: Isaac Sim for Humanoid Robotics',
      items: [
        'module3/intro',
        'module3/isaac-sim-fundamentals',
        'module3/rendering-materials',
        'module3/physics-simulation',
        'module3/synthetic-data-generation',
        'module3/isaac-ros-integration',
        'module3/bipedal-locomotion',
        'module3/advanced-features',
        'module3/summary',
        'module3/review-checklist'
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action Integration',
      items: [
        'module4/intro',
        'module4/multimodal-perception',
        'module4/llm-integration',
        'module4/vision-for-action',
        'module4/safety-validation',
        'module4/human-robot-interaction',
        'module4/sim-to-real-transfer',
        'module4/summary',
        'module4/review-checklist'
      ],
    }
  ],

  // But you can create a sidebar manually
  /*
  tutorialSidebar: [
    'intro',
    'hello',
    {
      type: 'category',
      label: 'Tutorial',
      items: ['tutorial-basics/create-a-document'],
    },
  ],
   */
};

export default sidebars;
