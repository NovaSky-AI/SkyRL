import Link from 'next/link';

export default function HomePage() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-24">
      <div className="z-10 max-w-5xl w-full items-center justify-center font-mono text-sm">
        <h1 className="text-4xl font-bold mb-8 text-center">SkyRL Documentation</h1>
        <p className="text-xl mb-8 text-center">
          A full-stack RL library designed for modularity and extensibility.
        </p>
        <div className="flex justify-center">
          <Link
            href="/docs"
            className="inline-block bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded text-lg"
          >
            View Documentation →
          </Link>
        </div>
      </div>
    </main>
  );
}
