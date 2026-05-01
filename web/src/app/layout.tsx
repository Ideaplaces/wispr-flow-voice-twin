import "./globals.css";

export const metadata = {
  title: "Voice Twin Explorer",
  description: "Browse a year of dictated thoughts laid out by semantic similarity.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
