import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useStore } from "@/store/useStore";

/**
 * Redirects to /auth if the user is not logged in.
 * Use in every protected page.
 */
export function useAuthGuard(): string {
  const router = useRouter();
  const userId = useStore((s) => s.userId);

  useEffect(() => {
    if (!userId) {
      router.replace("/auth");
    }
  }, [userId, router]);

  return userId || "";
}
